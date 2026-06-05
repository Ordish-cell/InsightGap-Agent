from datetime import UTC, datetime
from typing import Any

from src.web_app.core.config import settings


STEP_TITLES = {
    "home_intent": "判断需求",
    "planner": "生成计划",
    "context_builder": "构建上下文",
    "skill_matcher": "匹配 Skill",
    "research_agent": "执行研究",
    "rag_agent": "检索知识库",
    "artifact_agent": "生成产物",
    "tool_agent": "准备工具动作",
    "memory_agent": "写入记忆",
    "skill_agent": "沉淀 Skill",
    "evaluator": "评估结果",
    "final_response": "生成最终回答",
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
    return state
