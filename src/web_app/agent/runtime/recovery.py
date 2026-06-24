"""Evaluator-driven retry and recovery helpers."""

from __future__ import annotations

from typing import Any

from src.web_app.agent.runtime.state import AgentRuntimeState


RECOVERABLE_AGENT_NODES = {
    "research_agent",
    "rag_agent",
    "artifact_agent",
    "tool_agent",
    "memory_agent",
    "skill_agent",
}
DEFAULT_MAX_ATTEMPTS_PER_AGENT = 1
DEFAULT_MAX_TOTAL_ATTEMPTS = 6

_WARNING_TARGETS = {
    "rag_evidence_missing": "rag_agent",
    "rag_failed": "rag_agent",
    "tool_failed": "tool_agent",
    "artifact_missing": "artifact_agent",
    "memory_write_failed": "memory_agent",
    "skill_agent_failed": "skill_agent",
    "research_failed": "research_agent",
    "research_agent_failed": "research_agent",
}
_BLOCKED_WARNINGS = {
    "approval_pending",
    "tool_waiting_approval",
    "tool_denied",
}


def build_evaluator_recovery_decision(
    state: AgentRuntimeState,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Choose a single agent retry target from evaluator findings."""
    warnings = [str(item) for item in (warnings or state.get("final_warnings") or []) if item]
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    route = [str(item) for item in route_plan.get("route", []) or [] if item]
    attempts = _attempts(state)
    history = [item for item in state.get("evaluator_recovery_history", []) or [] if isinstance(item, dict)]

    blocked_reasons = _blocked_reasons(state, warnings)
    reason_by_target = _reason_by_target(state, warnings)
    target = _select_target(route, reason_by_target)
    max_per_agent = DEFAULT_MAX_ATTEMPTS_PER_AGENT
    max_total = DEFAULT_MAX_TOTAL_ATTEMPTS

    exhausted = False
    if target and attempts.get(target, 0) >= max_per_agent:
        blocked_reasons.append(f"max_attempts_for_agent:{target}")
        exhausted = True
    if len(history) >= max_total:
        blocked_reasons.append("max_total_recovery_attempts")
        exhausted = True

    should_retry = bool(target) and not blocked_reasons
    reason = reason_by_target.get(target or "", "")
    next_attempt = attempts.get(target or "", 0) + 1 if target else 0

    return {
        "mode": "evaluator_recovery",
        "should_retry": should_retry,
        "target": target or "",
        "reason": reason,
        "attempt": next_attempt,
        "max_attempts_per_agent": max_per_agent,
        "max_total_attempts": max_total,
        "attempts": attempts,
        "history_count": len(history),
        "warnings": warnings,
        "blocked_reasons": blocked_reasons,
        "exhausted": bool(exhausted or (target and blocked_reasons)),
    }


def apply_evaluator_recovery_decision(
    state: AgentRuntimeState,
    decision: dict[str, Any],
) -> None:
    state["evaluator_recovery_decision"] = decision
    target = str(decision.get("target") or "")
    if decision.get("should_retry") and target:
        attempts = _attempts(state)
        attempts[target] = int(attempts.get(target) or 0) + 1
        entry = {
            "target": target,
            "reason": decision.get("reason", ""),
            "attempt": attempts[target],
        }
        state["evaluator_recovery_attempts"] = attempts
        state.setdefault("evaluator_recovery_history", []).append(entry)
        state["evaluator_recovery_active"] = True
        state["evaluator_recovery_target"] = target
        state["evaluator_recovery_exhausted"] = False
        decision["attempts"] = dict(attempts)
        decision["attempt"] = attempts[target]
        decision["history_count"] = len(state.get("evaluator_recovery_history", []))
        return

    state["evaluator_recovery_active"] = False
    state["evaluator_recovery_target"] = None
    state["evaluator_recovery_exhausted"] = bool(decision.get("exhausted"))


def dispatch_after_evaluator(state: AgentRuntimeState) -> str:
    decision = state.get("evaluator_recovery_decision")
    if isinstance(decision, dict) and decision.get("should_retry"):
        target = str(decision.get("target") or "")
        if target in RECOVERABLE_AGENT_NODES:
            return target
    return "final_response"


def should_return_to_evaluator_after_recovery(state: AgentRuntimeState) -> bool:
    target = str(state.get("evaluator_recovery_target") or "")
    if not state.get("evaluator_recovery_active") or not target:
        return False
    completed = [str(item) for item in state.get("completed_nodes", []) or []]
    return bool(completed and completed[-1] == target)


def _reason_by_target(state: AgentRuntimeState, warnings: list[str]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for warning in warnings:
        target = _WARNING_TARGETS.get(warning)
        if target:
            reasons.setdefault(target, warning)

    latest_results: dict[str, dict[str, Any]] = {}
    for result in state.get("agent_results", []) or []:
        if not isinstance(result, dict):
            continue
        agent = str(result.get("agent") or "")
        if agent in RECOVERABLE_AGENT_NODES:
            latest_results[agent] = result
    for agent, result in latest_results.items():
        status = str(result.get("status") or "")
        if status not in {"failed", "timeout"}:
            continue
        reasons.setdefault(agent, f"{agent}_failed")
    return reasons


def _select_target(route: list[str], reason_by_target: dict[str, str]) -> str:
    for node in route:
        if node in reason_by_target:
            return node
    for node in (
        "rag_agent",
        "tool_agent",
        "artifact_agent",
        "memory_agent",
        "skill_agent",
        "research_agent",
    ):
        if node in reason_by_target:
            return node
    return ""


def _blocked_reasons(state: AgentRuntimeState, warnings: list[str]) -> list[str]:
    blocked: list[str] = []
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    risk_level = str(route_plan.get("risk_level") or "")
    status = str(state.get("status") or "")
    if risk_level == "L4":
        blocked.append("risk_level:L4")
    if status == "waiting_approval":
        blocked.append("waiting_approval")
    if state.get("approval_payload") or state.get("approval_required"):
        blocked.append("approval_pending")
    if state.get("pending_approval_id") or state.get("pending_tool_call_id"):
        blocked.append("pending_tool_or_approval")
    for warning in warnings:
        if warning in _BLOCKED_WARNINGS and warning not in blocked:
            blocked.append(warning)
    return blocked


def _attempts(state: AgentRuntimeState) -> dict[str, int]:
    value = state.get("evaluator_recovery_attempts")
    if not isinstance(value, dict):
        return {}
    return {str(key): int(count or 0) for key, count in value.items()}
