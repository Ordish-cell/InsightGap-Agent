"""Durable control requests for ordinary text chat (one application worker)."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select

from src.web_app.agent.runtime.chat_control import capabilities, transition_lock
from src.web_app.agent.runtime.event_ledger import publish_event
from src.web_app.models.orm import AgentChatMessage, AgentConversation, AgentRun, AgentRunControl

ACTIVE_STATUSES = {"created", "running", "resuming", "waiting_approval", "paused", "queued"}


class ChatControlError(ValueError):
    def __init__(self, code, message, status=409, **details):
        super().__init__(message)
        self.status = status
        self.detail = {"code": code, "message": message, **details}


def ensure_conversation_idle(db, user_id, conversation_id):
    if not conversation_id:
        return
    db.execute(select(AgentConversation.id).where(
        AgentConversation.user_id == user_id, AgentConversation.conversation_id == conversation_id,
    ).with_for_update()).first()
    active = db.execute(select(AgentRun).where(
        AgentRun.user_id == user_id, AgentRun.conversation_id == conversation_id,
        AgentRun.status.in_(ACTIVE_STATUSES),
    ).order_by(AgentRun.id.desc())).scalars().first()
    if active:
        raise ChatControlError("CONVERSATION_BUSY", "当前会话仍有任务，请等待或使用聊天插话。", latest_run_id=active.id)


def control_response(command):
    return {"client_command_id": command.client_command_id, "run_id": command.run_id,
            "successor_run_id": command.successor_run_id, "kind": command.kind,
            "status": command.status, "error": command.error_message}


def request_control(db, user_id, run_id, kind, client_command_id, text=""):
    from src.web_app.services.agent_run_task_manager import agent_run_task_manager
    with transition_lock:
        run = db.execute(select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id)
                         .execution_options(populate_existing=True)).scalar_one_or_none()
        if not run:
            raise ChatControlError("RUN_NOT_FOUND", "任务不存在。", 404)
        existing = db.execute(select(AgentRunControl).where(
            AgentRunControl.user_id == user_id, AgentRunControl.client_command_id == client_command_id,
        )).scalar_one_or_none()
        if existing:
            if (existing.run_id, existing.kind, existing.text) != (run_id, kind, text):
                raise ChatControlError("COMMAND_ID_CONFLICT", "此请求 ID 已用于其他操作。")
            return control_response(existing)
        conversation = db.execute(select(AgentConversation).where(
            AgentConversation.conversation_id == run.conversation_id,
            AgentConversation.user_id == user_id, AgentConversation.status.not_in(["deleted", "deleting"]),
        ).with_for_update()).scalar_one_or_none()
        if not conversation:
            raise ChatControlError("CONVERSATION_NOT_FOUND", "会话不存在。", 404)
        successor = db.execute(select(AgentRun).where(AgentRun.supersedes_run_id == run_id)
                               .order_by(AgentRun.id.desc())).scalars().first()
        if successor:
            raise ChatControlError("RUN_SUPERSEDED", "任务已接续，请刷新后向最新任务发送。", latest_run_id=successor.id)
        pending = db.execute(select(AgentRunControl).where(
            AgentRunControl.run_id == run_id, AgentRunControl.status == "accepted",
        )).scalars().first()
        if pending:
            raise ChatControlError("CONTROL_PENDING", "正在处理上一条控制请求。")
        terminal = run.status in {"completed", "interrupted", "failed"}
        if not terminal and (not capabilities(run)["can_interrupt"] or not agent_run_task_manager.is_running(run_id)):
            raise ChatControlError("CHAT_CONTROL_UNAVAILABLE", "当前阶段不支持普通聊天打断。")
        if terminal and kind == "steer":
            ensure_conversation_idle(db, user_id, run.conversation_id)

        command = AgentRunControl(user_id=user_id, run_id=run_id, kind=kind,
                                  client_command_id=client_command_id, text=text, status="accepted", error_message="")
        db.add(command)
        if kind == "steer":
            # Reuse the frozen public model context. Secrets are resolved at execution.
            previous = run.graph_state or {}
            successor = AgentRun(user_id=user_id, conversation_id=run.conversation_id,
                                 thread_id=run.thread_id, mode=run.mode, run_type=run.run_type,
                                 status="queued", user_input=text, supersedes_run_id=run_id,
                                 chat_control_phase="queued", graph_state={})
            db.add(successor)
            db.flush()
            successor.graph_state = {"thread_id": f"run:{successor.id}", "conversation_id": run.conversation_id,
                                     "conversation_thread_id": run.thread_id, "source": "chat_steer",
                                     "model_context": previous.get("model_context", {}),
                                     "runtime_version": previous.get("runtime_version", 2),
                                     "model_config_id": previous.get("model_config_id"), "page_context": {}}
            for role, content, status in (("user", text, "completed"), ("assistant", "", "queued")):
                db.add(AgentChatMessage(message_id=str(uuid4()), user_id=user_id, conversation_id=run.conversation_id,
                                       thread_id=run.thread_id, run_id=successor.id, role=role, content=content,
                                       status=status, metadata_json={"supersedes_run_id": run_id, "interaction_version": 2}))
            command.successor_run_id = successor.id
            conversation.last_run_id = successor.id
            conversation.last_message_preview = text[:400]
            conversation.message_count += 2
            conversation.last_active_at = datetime.now()
        if not terminal:
            run.chat_control_phase = "stopping"
        db.commit()
        # No await occurs between persistence and fencing the active execution.
        if not terminal:
            agent_run_task_manager.fence(run_id)
        publish_event(db, None, run_id, "control_accepted", control_response(command),
                      user_id=user_id, thread_id=run.thread_id)
        agent_run_task_manager.schedule_control(command.id)
        return control_response(command)


def continuation_context(db, run):
    """Pin original input and successive corrections outside GSSC selection."""
    if not run.supersedes_run_id:
        return ""
    chain = []
    previous_id = run.supersedes_run_id
    seen = set()
    while previous_id and previous_id not in seen:
        seen.add(previous_id)
        previous = db.execute(select(AgentRun).where(AgentRun.id == previous_id, AgentRun.user_id == run.user_id,
                                                    AgentRun.conversation_id == run.conversation_id)).scalar_one_or_none()
        if not previous:
            break
        chain.append(previous)
        previous_id = previous.supersedes_run_id
    chain.reverse()
    lines = ["[Chat continuation — conversation data]",
             "The user interrupted a reply. Apply the latest explicit corrections, retaining compatible earlier requirements. "
             "If the latest input changes topic, answer the new topic. Partial assistant text is unfinished, not verified evidence."]
    for previous in chain:
        lines.append(f"User: {previous.user_input}")
    # Only partial output is truncated; original input and corrections remain intact.
    if chain:
        lines.append(f"Assistant (unfinished): {(chain[-1].final_answer or '')[:4000]}")
    lines.append(f"Latest user input: {run.user_input}")
    return "\n".join(lines)


def recover_chat_controls(session_factory):
    """Startup only: never replay work or touch research/approval runs."""
    with session_factory() as db:
        runs = db.execute(select(AgentRun).where(
            AgentRun.chat_control_phase.in_({"enabled", "stopping", "queued"}),
            AgentRun.status.in_(ACTIVE_STATUSES),
        )).scalars().all()
        from src.web_app.services.agent_run_task_manager import AgentRunTaskManager
        manager = AgentRunTaskManager()
        for run in runs:
            manager._mark_terminal(db, run.user_id, run.id, "interrupted", "服务重启，聊天已中断；消息已保留，请重试。")
        commands = db.execute(select(AgentRunControl).where(AgentRunControl.status == "accepted")).scalars().all()
        for command in commands:
            command.status = "failed"
            command.error_message = "服务重启，控制未完成；消息已保留，请重试。"
        db.commit()
