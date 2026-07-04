from datetime import UTC, datetime
from typing import Any

from src.web_app.core.config import settings

# Module-level stream queue — set by AgentRuntime.run() before graph execution,
# read by _emit_status_sse(). Avoids putting a non-serializable queue into
# LangGraph state (which would break checkpointing).
_stream_queue: Any = None


def set_status_stream_queue(queue: Any) -> None:
    global _stream_queue
    _stream_queue = queue


def clear_status_stream_queue() -> None:
    global _stream_queue
    _stream_queue = None


STEP_TITLES = {
    "home_intent": "\u5224\u65ad\u9700\u6c42",
    "planner": "\u751f\u6210\u8ba1\u5212",
    "context_builder": "\u6784\u5efa\u4e0a\u4e0b\u6587",
    "skill_matcher": "\u5339\u914d Skill",
    "research_agent": "\u6267\u884c\u7814\u7a76",
    "rag_agent": "\u68c0\u7d22\u77e5\u8bc6\u5e93",
    "artifact_agent": "\u751f\u6210\u4ea7\u7269",
    "tool_agent": "\u51c6\u5907\u5de5\u5177\u52a8\u4f5c",
    "memory_agent": "\u5199\u5165\u8bb0\u5fc6",
    "skill_agent": "\u6c89\u6dc0 Skill",
    "evaluator": "\u8bc4\u4f30\u7ed3\u679c",
    "final_response": "\u751f\u6210\u6700\u7ec8\u56de\u7b54",
}

STEP_THOUGHTS = {
    "home_intent": "\u6211\u5148\u5224\u65ad\u8bf7\u6c42\u7c7b\u578b\u3001\u98ce\u9669\u7b49\u7ea7\u548c\u53ef\u7528\u80fd\u529b\u3002",
    "planner": "\u6211\u628a\u4efb\u52a1\u62c6\u6210\u53ef\u6267\u884c\u6b65\u9aa4\u3002",
    "context_builder": "\u6211\u6536\u96c6\u4f1a\u8bdd\u3001\u8bb0\u5fc6\u548c\u9875\u9762\u4e0a\u4e0b\u6587\u3002",
    "skill_matcher": "\u6211\u68c0\u67e5\u662f\u5426\u6709\u53ef\u590d\u7528 Skill\u3002",
    "research_agent": "\u6211\u6267\u884c\u7814\u7a76\u5e76\u6574\u7406\u8bc1\u636e\u3002",
    "rag_agent": "\u6211\u68c0\u7d22\u77e5\u8bc6\u5e93\u5e76\u5408\u5e76\u76f8\u5173\u4f9d\u636e\u3002",
    "artifact_agent": "\u6211\u751f\u6210\u53ef\u4fdd\u5b58\u6216\u590d\u7528\u7684\u4ea7\u7269\u3002",
    "tool_agent": "\u6211\u68c0\u67e5\u5de5\u5177\u52a8\u4f5c\u548c\u5ba1\u6279\u8981\u6c42\u3002",
    "memory_agent": "\u6211\u5224\u65ad\u662f\u5426\u9700\u8981\u5199\u5165\u957f\u671f\u8bb0\u5fc6\u3002",
    "skill_agent": "\u6211\u5224\u65ad\u662f\u5426\u503c\u5f97\u6c89\u6dc0\u4e3a Skill\u3002",
    "evaluator": "\u6211\u68c0\u67e5\u6267\u884c\u7ed3\u679c\u548c\u98ce\u9669\u72b6\u6001\u3002",
    "final_response": "\u6211\u8c03\u7528\u6700\u7ec8\u56de\u590d\u6a21\u578b\uff0c\u628a\u8fc7\u7a0b\u7ed3\u679c\u6574\u7406\u6210\u4f60\u80fd\u76f4\u63a5\u4f7f\u7528\u7684\u56de\u7b54\u3002",
}


def append_status_step(
    state: dict[str, Any],
    *,
    key: str,
    node_name: str,
    status: str = "completed",
    detail: str = "",
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not settings.agent_langgraph_status_enabled:
        return state

    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    current = state.setdefault(
        "langgraphstatus",
        {
            "run_id": str(state.get("run_id", "")),
            "thread_id": state.get("thread_id", ""),
            "status": state.get("status", "running"),
            "current_node": node_name,
            "intent": (state.get("route_plan") or {}).get("intent", ""),
            "risk_level": (state.get("route_plan") or {}).get("risk_level", "L0"),
            "needs_approval": bool(state.get("approval_required", False)),
            "summary": detail,
            "steps": [],
        },
    )
    steps = list(current.get("steps", []))
    steps = [item for item in steps if item.get("key") != key]
    item = {
        "key": key,
        "title": STEP_TITLES.get(key, key),
        "status": status,
        "node_name": node_name,
        "detail": detail,
        "model": model,
        "started_at": now,
        "completed_at": now if status in {"completed", "failed", "waiting_approval"} else None,
    }
    if extra:
        item.update(extra)
    _add_visible_react_fields(item, key)
    steps.append(item)
    current.update(
        {
            "run_id": str(state.get("run_id", "")),
            "thread_id": state.get("thread_id", ""),
            "status": state.get("status", "running"),
            "current_node": node_name,
            "intent": (state.get("route_plan") or {}).get("intent") or (state.get("home_intent") or {}).get("intent", ""),
            "risk_level": (state.get("route_plan") or {}).get("risk_level") or (state.get("home_intent") or {}).get("risk_level", "L0"),
            "needs_approval": bool(state.get("approval_required") or (state.get("home_intent") or {}).get("needs_approval")),
            "summary": detail,
            "steps": steps[-settings.agent_langgraph_status_max_steps :],
        }
    )
    state["langgraphstatus"] = current

    # ── Emit SSE event so frontend shows this step in real-time ──
    _emit_status_sse(state, key, item, status)

    return state


def _emit_status_sse(state: dict[str, Any], key: str, item: dict[str, Any], status: str) -> None:
    """Push a status_step SSE event via the stream queue so the frontend
    renders each pipeline step as it completes, not just on refresh."""
    stream_queue = _stream_queue  # module-level, set by AgentRuntime.run()
    if stream_queue is None:
        return
    try:
        payload = {
            "run_id": state.get("run_id"),
            "thread_id": state.get("thread_id", ""),
            "event_type": "status_step",
            "node_name": key,
            "visibility": "user",
            "display_channel": "progress",
            "payload": {
                "key": key,
                "title": item.get("title", key),
                "status": status,
                "detail": item.get("detail", ""),
                "thought": item.get("thought", item.get("title", key)),
            },
            "created_at": item.get("started_at", ""),
        }
        stream_queue.put_nowait({"event": "status_step", "data": payload})
    except Exception:
        pass  # never let SSE failures break the pipeline


def _add_visible_react_fields(item: dict[str, Any], key: str) -> None:
    title = str(item.get("title") or STEP_TITLES.get(key, key))
    detail = str(item.get("detail") or item.get("summary") or item.get("reason_summary") or "")
    if _looks_mojibake(detail):
        detail = ""
    thought = str(item.get("thought") or STEP_THOUGHTS.get(key) or title)
    if detail and detail not in thought:
        thought = f"{thought} {detail}"
    item.setdefault("thought", thought)
    item.setdefault("action", title)
    item.setdefault("observation", detail or _status_label(str(item.get("status") or "completed")))
    item.setdefault("next_action", _next_action_label(str(item.get("status") or "completed")))


def _status_label(status: str) -> str:
    if status == "running":
        return "\u6b63\u5728\u5904\u7406"
    if status == "failed":
        return "\u5931\u8d25"
    if status == "waiting_approval":
        return "\u7b49\u5f85\u5ba1\u6279"
    return "\u5df2\u5b8c\u6210"


def _next_action_label(status: str) -> str:
    if status == "failed":
        return "\u6682\u505c\u540e\u7eed\u6267\u884c\uff0c\u4f18\u5148\u5904\u7406\u5931\u8d25\u539f\u56e0\u3002"
    if status == "waiting_approval":
        return "\u7b49\u5f85\u7528\u6237\u5ba1\u6279\u540e\u7ee7\u7eed\u3002"
    if status == "running":
        return "\u7ee7\u7eed\u63a8\u8fdb\u5f53\u524d\u8282\u70b9\u3002"
    return "\u8fd9\u4e00\u6b65\u5df2\u5b8c\u6210\uff0c\u7ed3\u679c\u4f1a\u8fdb\u5165\u540e\u7eed\u4e0a\u4e0b\u6587\u3002"


def _looks_mojibake(value: str) -> bool:
    return any(marker in value for marker in ("鍒", "璇", "宸", "鐮", "妫", "鈫", "锛", "€", ""))
