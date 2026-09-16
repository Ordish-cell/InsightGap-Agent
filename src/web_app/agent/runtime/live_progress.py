"""Real node lifetimes and evidence-backed results, independent of UI wording."""
import asyncio
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

current_step = ContextVar("current_live_step", default=None)
step_started_at = ContextVar("live_step_started_at", default=None)
NAMES = {
    "permission_guard": "检查操作权限",
    "bootstrap_context": "读取会话上下文",
    "supervisor": "分析当前结果",
    "capability": "处理请求",
    "deep_research": "深入研究",
    "tool_runtime": "执行工具",
    "document_read": "读取文档",
}


async def run_node(name, node, db, state, args, kwargs):
    if db is None or state.get("interaction_version") != 2:
        return await node(state, *args, **kwargs)
    from src.web_app.agent.runtime.event_ledger import publish_event
    step_id = uuid4().hex
    parent = current_step.get()
    marker = current_step.set(step_id)
    timing_marker = step_started_at.set(datetime.now(UTC).replace(tzinfo=None))
    started = perf_counter()
    base = {"step_id": step_id, "parent_step_id": parent, "display_name": NAMES.get(name, name)}
    def emit(kind, **extra):
        publish_event(db, None, state["run_id"], kind, {**base, **extra}, node_name=name,
                      user_id=state.get("user_id"), thread_id=state.get("thread_id"))
    try:
        emit("node_started", status="running")
        try:
            result = await node(state, *args, **kwargs)
            from src.web_app.agent.runtime.chat_control import check_active
            check_active(state["run_id"])
        except asyncio.CancelledError:
            emit("node_cancelled", status="cancelled", elapsed_ms=(perf_counter() - started) * 1000)
            raise
        except BaseException as exc:
            from langgraph.errors import GraphInterrupt
            if isinstance(exc, GraphInterrupt):
                emit("node_paused", status="waiting_approval")
            else:
                emit("node_failed", status="failed", elapsed_ms=(perf_counter() - started) * 1000)
            raise
        status = result.get("status")
        event_type = "node_paused" if status == "waiting_approval" else "node_failed" if status == "failed" else "node_completed"
        emit(event_type, status=status if status in {"waiting_approval", "failed"} else "completed", elapsed_ms=(perf_counter() - started) * 1000)
        return result
    finally:
        step_started_at.reset(timing_marker)
        current_step.reset(marker)
