"""Single-process chat cancellation boundary, shared by API and execution.

The lock protects short synchronous transitions only; never hold it across await.
The mutable token also fences child tasks/threads which outlive cancellation.
"""
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import RLock

from sqlalchemy import select

from src.web_app.models.orm import AgentRun

transition_lock = RLock()


def serialized_transition(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with transition_lock:
            return fn(*args, **kwargs)
    return wrapped


@dataclass
class ChatExecution:
    run_id: int
    cancelled: bool = False
    disabled: bool = False


execution: ContextVar[ChatExecution | None] = ContextVar("chat_execution", default=None)


def check_active(run_id=None):
    token = execution.get()
    if token and (run_id is None or token.run_id == run_id) and token.cancelled:
        raise asyncio.CancelledError("Chat execution superseded or stopped")


def capabilities(run):
    enabled = run.status in {"created", "running"} and run.chat_control_phase == "enabled"
    return {"can_interrupt": enabled, "can_steer": enabled,
            "supersedes_run_id": run.supersedes_run_id}


def disable_control(db, run_id):
    token = execution.get()
    if not token or token.run_id != run_id:
        return
    from src.web_app.agent.runtime.event_ledger import publish_event
    with transition_lock:
        check_active(run_id)
        if token.disabled:
            return
        run = db.execute(select(AgentRun).where(AgentRun.id == run_id).execution_options(populate_existing=True)).scalar_one()
        if run.chat_control_phase == "stopping":
            token.cancelled = True
            check_active(run_id)
        token.disabled = True
        run.chat_control_phase = "disabled"
        db.commit()
        publish_event(db, None, run_id, "run_capabilities", capabilities(run), user_id=run.user_id, thread_id=run.thread_id)


def controlled_node(name, node, db):
    @wraps(node)
    async def wrapped(state, *args, **kwargs):
        token = execution.get()
        if token:
            with transition_lock:
                check_active()
                from src.web_app.services.deletion_guard import check_conversation
                if db is not None and state.get("conversation_id"):
                    check_conversation(db, state.get("user_id"), state["conversation_id"])
                intent = (state.get("route_plan") or {}).get("intent")
                if name in {"research_agent", "rag_agent", "tool_agent", "artifact_agent", "memory_agent", "skill_agent"} or (intent and intent != "chat"):
                    disable_control(db, state["run_id"])
        from src.web_app.agent.runtime.live_progress import run_node
        result = await run_node(name, node, db, state, args, kwargs)
        check_active()
        return result
    return wrapped
