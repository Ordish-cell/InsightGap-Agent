"""Process-local ownership for Agent run execution tasks."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from src.web_app.agent.runtime.chat_control import ChatExecution, execution, transition_lock
from src.web_app.models.orm import AgentRun, AgentRunControl, AgentEvent

from src.web_app.agent.runtime.event_ledger import publish_event
from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository, AgentRunRepository
from src.web_app.db.session import SessionLocal


logger = logging.getLogger(__name__)
CHAT_CANCEL_TIMEOUT_SECONDS = 2.0


class AgentRunTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._tokens: dict[int, ChatExecution] = {}
        self._controls: dict[int, asyncio.Task] = {}

    def is_running(self, run_id: int) -> bool:
        task = self._tasks.get(run_id)
        return bool(task and not task.done())

    def start(self, run_id: int, user_id: int, *, payload: dict[str, Any] | None = None, resume: bool = False) -> bool:
        if self.is_running(run_id):
            return False
        task = asyncio.create_task(self._run(run_id, user_id, payload or {}, resume=resume))
        self._tasks[run_id] = task
        task.add_done_callback(lambda completed, rid=run_id: self._discard(rid, completed))
        return True

    async def _run(self, run_id: int, user_id: int, payload: dict[str, Any], *, resume: bool) -> None:
        from src.web_app.services.agent_service import execute_prepared_run, resume_run_after_approval

        token = self._tokens.setdefault(run_id, ChatExecution(run_id)) if payload.get("_chat_managed") else None
        context_token = execution.set(token)
        watcher = asyncio.create_task(self._poll_controls(run_id)) if token else None
        with SessionLocal() as db:
            try:
                if token and token.cancelled:
                    raise asyncio.CancelledError()
                if resume:
                    await resume_run_after_approval(db, user_id, run_id)
                else:
                    await execute_prepared_run(db, user_id, run_id, payload)
            except asyncio.CancelledError:
                db.rollback()
                # Never finalize using the cancelled execution's session or token.
                execution.set(None)
                with SessionLocal() as terminal_db:
                    self._mark_terminal(terminal_db, user_id, run_id, "interrupted", "聊天已中断，已保留输出。" if token and token.cancelled else "Application shutdown interrupted the run.")
                raise
            except Exception as exc:
                logger.exception("Agent background task failed run_id=%s", run_id)
                db.rollback()
                self._mark_terminal(db, user_id, run_id, "failed", str(exc))
            finally:
                if watcher:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
                execution.reset(context_token)

    def _mark_terminal(self, db, user_id: int, run_id: int, status: str, error: str) -> None:
        run_repo = AgentRunRepository(db)
        run = run_repo.get_by_user(user_id, run_id)
        if not run or run.status in {"completed", "failed", "interrupted"}:
            return
        event_type = "run_interrupted" if status == "interrupted" else "run_failed"
        chunks = db.execute(select(AgentEvent).where(AgentEvent.run_id == run_id, AgentEvent.user_id == user_id,
                                                     AgentEvent.event_type.in_({"answer_delta", "answer_completed"})).order_by(AgentEvent.id)).scalars()
        partial = ""
        for event in chunks:
            payload = event.payload_json or {}
            partial = str(payload.get("answer", partial)) if event.event_type == "answer_completed" else partial + str(payload.get("text", ""))
        run.chat_control_phase = "interrupted"
        run_repo.update(run, status=status, error_message=error, completed_at=datetime.now(), final_answer=partial, result_summary=partial)
        messages = AgentChatMessageRepository(db).list_by_conversation(user_id, run.conversation_id)
        assistant = next((item for item in messages if item.run_id == run_id and item.role == "assistant"), None)
        if assistant:
            AgentChatMessageRepository(db).update(assistant, status=status, content=partial, error_message=error,
                                                  metadata_json={**(assistant.metadata_json or {}), "interrupted": status == "interrupted"})
        publish_event(
            db,
            None,
            run_id,
            event_type,
            {"status": status, "error": error, "run_id": run_id, "answer": partial,
             "message_id": assistant.message_id if assistant else None, "can_interrupt": False, "can_steer": False},
            user_id=user_id,
            thread_id=run.thread_id,
        )

    def _discard(self, run_id: int, task: asyncio.Task[None]) -> None:
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)
            self._tokens.pop(run_id, None)

    def fence(self, run_id):
        self._tokens.setdefault(run_id, ChatExecution(run_id)).cancelled = True

    def schedule_control(self, command_id):
        if command_id not in self._controls:
            task = asyncio.create_task(self._apply_control(command_id))
            self._controls[command_id] = task
            task.add_done_callback(lambda _: self._controls.pop(command_id, None))

    async def _poll_controls(self, run_id):
        execution.set(None)
        while True:
            await asyncio.sleep(0.25)
            with SessionLocal() as db:
                pending = list(db.execute(select(AgentRunControl.id).where(
                    AgentRunControl.run_id == run_id, AgentRunControl.status == "accepted",
                )).scalars())
            for command_id in pending:
                self.fence(run_id)
                self.schedule_control(command_id)

    async def _apply_control(self, command_id):
        from src.web_app.services.chat_control_service import control_response
        execution.set(None)
        try:
            with SessionLocal() as db:
                command = db.get(AgentRunControl, command_id)
                if not command or command.status != "accepted":
                    return
                run_id = command.run_id
            task = self._tasks.get(run_id)
            if task and not task.done():
                task.cancel()
                done, _ = await asyncio.wait({task}, timeout=CHAT_CANCEL_TIMEOUT_SECONDS)
                if not done:
                    raise TimeoutError("停止生成超时，消息已保留，请重试。")
            with transition_lock, SessionLocal() as db:
                command = db.get(AgentRunControl, command_id)
                if not command or command.status != "accepted":
                    return
                run = db.get(AgentRun, command.run_id)
                if run.status not in {"completed", "interrupted", "failed"}:
                    # A task cancelled before its first instruction cannot run its finally.
                    self._mark_terminal(db, run.user_id, run.id, "interrupted", "聊天已停止。")
                successor = db.get(AgentRun, command.successor_run_id) if command.successor_run_id else None
                if successor:
                    successor.status = "created"
                    successor.chat_control_phase = "enabled"
                command.status = "applied"
                db.commit()
                target = successor or run
                publish_event(db, None, target.id, "control_applied", control_response(command), user_id=target.user_id, thread_id=target.thread_id)
                if successor:
                    self.start(successor.id, successor.user_id, payload={"_chat_managed": True, "source": "chat_steer"})
        except Exception as exc:
            logger.exception("Chat control failed command_id=%s", command_id)
            with SessionLocal() as db:
                command = db.get(AgentRunControl, command_id)
                if not command:
                    return
                command.status = "failed"
                command.error_message = str(exc)
                db.commit()
                target_id = command.successor_run_id or command.run_id
                publish_event(db, None, target_id, "control_failed", control_response(command), user_id=command.user_id)
                if command.successor_run_id:
                    self._mark_terminal(db, command.user_id, target_id, "failed", str(exc))

    async def shutdown(self) -> None:
        controls = list(self._controls.values())
        for control in controls:
            control.cancel()
        if controls:
            await asyncio.gather(*controls, return_exceptions=True)
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


agent_run_task_manager = AgentRunTaskManager()
