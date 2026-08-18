"""Process-local ownership for Agent run execution tasks."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.web_app.agent.runtime.event_ledger import publish_event
from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository, AgentRunRepository
from src.web_app.db.session import SessionLocal


logger = logging.getLogger(__name__)


class AgentRunTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[int, asyncio.Task[None]] = {}

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

        with SessionLocal() as db:
            try:
                if resume:
                    await resume_run_after_approval(db, user_id, run_id)
                else:
                    await execute_prepared_run(db, user_id, run_id, payload)
            except asyncio.CancelledError:
                self._mark_terminal(db, user_id, run_id, "interrupted", "Application shutdown interrupted the run.")
                raise
            except Exception as exc:
                logger.exception("Agent background task failed run_id=%s", run_id)
                db.rollback()
                self._mark_terminal(db, user_id, run_id, "failed", str(exc))

    def _mark_terminal(self, db, user_id: int, run_id: int, status: str, error: str) -> None:
        run_repo = AgentRunRepository(db)
        run = run_repo.get_by_user(user_id, run_id)
        if not run or run.status in {"completed", "failed", "interrupted"}:
            return
        event_type = "run_interrupted" if status == "interrupted" else "run_failed"
        run_repo.update(run, status=status, error_message=error, completed_at=datetime.now())
        messages = AgentChatMessageRepository(db).list_by_conversation(user_id, run.conversation_id)
        assistant = next((item for item in messages if item.run_id == run_id and item.role == "assistant"), None)
        if assistant:
            AgentChatMessageRepository(db).update(assistant, status=status, error_message=error)
        publish_event(
            db,
            None,
            run_id,
            event_type,
            {"status": status, "error": error, "run_id": run_id},
            user_id=user_id,
            thread_id=run.thread_id,
        )

    def _discard(self, run_id: int, task: asyncio.Task[None]) -> None:
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


agent_run_task_manager = AgentRunTaskManager()
