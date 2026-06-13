"""StateDelta merge helpers for compatible LangGraph node results."""

from __future__ import annotations

from typing import Any

from src.web_app.agent.runtime.schemas import AgentResult, NodeResult, StateDelta, append_agent_result
from src.web_app.agent.runtime.state import AgentRuntimeState, mark_completed


_PROTECTED_UPDATE_FIELDS = {
    "status",
    "approval_payload",
    "pending_tool_call_id",
    "pending_approval_id",
    "pending_tool_name",
    "pending_tool_args",
    "resume_token",
    "final_payload",
    "final_output",
}


def apply_state_delta(
    state: AgentRuntimeState,
    delta: StateDelta | dict[str, Any],
    *,
    allow_protected: bool = False,
) -> AgentRuntimeState:
    payload = delta.model_dump() if isinstance(delta, StateDelta) else StateDelta(**delta).model_dump()

    for key, value in payload.get("updates", {}).items():
        if not allow_protected and _is_protected_update_key(key):
            continue
        state[key] = value

    for key, values in payload.get("append", {}).items():
        if not isinstance(values, list):
            values = [values]
        state.setdefault(key, []).extend(values)

    for warning in payload.get("warnings", []) or []:
        state.setdefault("node_warnings", []).append(warning)

    for event in payload.get("events", []) or []:
        state.setdefault("events", []).append(event)

    agent_result = payload.get("agent_result")
    if agent_result:
        append_agent_result(state, agent_result)

    completed_node = payload.get("completed_node")
    if completed_node:
        mark_completed(state, str(completed_node))

    return state


def append_node_result(state: AgentRuntimeState, result: NodeResult | dict[str, Any]) -> AgentRuntimeState:
    payload = result.model_dump() if isinstance(result, NodeResult) else NodeResult(**result).model_dump()
    state.setdefault("node_results", []).append(payload)
    return state


def record_node_result(
    state: AgentRuntimeState,
    *,
    node: str,
    status: str = "ok",
    delta: StateDelta | dict[str, Any] | None = None,
    summary: str = "",
    elapsed_ms: int | None = None,
) -> AgentRuntimeState:
    node_result = NodeResult(
        node=node,
        status=status,  # type: ignore[arg-type]
        delta=delta if isinstance(delta, StateDelta) else StateDelta(**(delta or {})),
        summary=summary,
        elapsed_ms=elapsed_ms,
    )
    return append_node_result(state, node_result)


def latest_agent_result(state: AgentRuntimeState, agent: str) -> dict[str, Any] | None:
    for result in reversed(list(state.get("agent_results") or [])):
        if isinstance(result, dict) and result.get("agent") == agent:
            return result
    return None


def record_agent_node_result(
    state: AgentRuntimeState,
    *,
    node: str,
    updates: dict[str, Any],
    summary: str = "",
    elapsed_ms: int | None = None,
    status: str | None = None,
) -> AgentRuntimeState:
    agent_result = latest_agent_result(state, node)
    node_status = status or _node_status_from_agent_result(agent_result)
    delta = StateDelta(
        updates=updates,
        agent_result=agent_result,
        metadata={"source": "formal_agent_node", "agent": node},
    )
    return record_node_result(
        state,
        node=node,
        status=node_status,
        delta=delta,
        summary=summary or (agent_result or {}).get("summary", ""),
        elapsed_ms=elapsed_ms,
    )


def _node_status_from_agent_result(agent_result: dict[str, Any] | None) -> str:
    if not agent_result:
        return "ok"
    status = str(agent_result.get("status") or "ok")
    if status in {"ok", "failed", "skipped", "needs_approval", "denied", "timeout"}:
        return status
    return "ok"


def _is_protected_update_key(key: str) -> bool:
    return key in _PROTECTED_UPDATE_FIELDS or key.startswith("pending_")
