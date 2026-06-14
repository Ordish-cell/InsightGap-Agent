"""Shadow-only replanner observations.

P9A does not replan. It only summarizes whether a future replanner might have
work to do, based on signals already produced by the runtime.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.core.config import get_settings


_FAILED_STATUSES = {"failed", "timeout", "denied"}
_REPLAN_CONTRACT_PREFIXES = (
    "missing_node_result:",
    "agent_node_status_mismatch:",
    "route_node_not_completed:",
    "agent_result_without_node_result:",
)
_UNSAFE_ROUTE_AGENTS = {
    "tool_agent",
    "artifact_agent",
    "memory_agent",
    "skill_agent",
    "research_agent",
}
_UNSAFE_INTENTS = {
    "tool",
    "tool.email",
    "tool.local_file",
    "tool.web_search",
    "tool.browser",
    "tool.comment",
    "tool.form_submit",
    "tool.shell_readonly",
    "tool.shell_write",
    "tool.dangerous",
    "system.time",
    "system.calc",
    "system.unit_convert",
    "system.uuid",
    "system.hash",
    "artifact",
    "memory",
    "memory_confirm",
    "skill",
    "research",
    "feed_research",
    "mixed",
}
_CANDIDATE_ALLOWED_INTENTS = {"chat", "rag"}
_CANDIDATE_ALLOWED_ROUTE_NODES = {"rag_agent", "final_response"}
_CONTROL_ALLOWED_NODES = {"rag_agent", "final_response"}


def build_replanner_shadow_report(state: dict[str, Any]) -> dict[str, Any]:
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    current_route = [str(item) for item in route_plan.get("route", []) or []]
    intent = str(route_plan.get("intent") or state.get("route") or "chat")
    risk_level = str(route_plan.get("risk_level") or "L0")

    trigger_sources: list[str] = []
    replan_reasons: list[str] = []
    warnings: list[str] = []

    _collect_supervisor_signals(state, trigger_sources, replan_reasons)
    _collect_contract_signals(state, trigger_sources, replan_reasons)
    _collect_agent_failure_signals(state, trigger_sources, replan_reasons)
    _collect_shadow_warning_signals(state, trigger_sources, replan_reasons)

    blocked_reasons = _blocked_reasons(state, intent, risk_level, current_route)
    if blocked_reasons:
        _append_once(warnings, "replanner_shadow_blocked")

    replan_recommended = bool(replan_reasons) and not blocked_reasons
    confidence = _confidence(replan_reasons, blocked_reasons)
    suggested_route = _suggested_route(current_route, replan_recommended)
    suggested_actions = _suggested_actions(replan_reasons, blocked_reasons, suggested_route)

    report = {
        "mode": "shadow_only",
        "replan_recommended": replan_recommended,
        "confidence": confidence,
        "trigger_sources": trigger_sources,
        "replan_reasons": replan_reasons,
        "blocked_reasons": blocked_reasons,
        "current_route": current_route,
        "suggested_route": suggested_route,
        "suggested_actions": suggested_actions,
        "safety_level": risk_level,
    }
    return {
        "replanner_shadow_report": report,
        "replanner_shadow_warnings": warnings,
    }


def build_replanner_candidate_plan(state: dict[str, Any]) -> dict[str, Any]:
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    shadow = state.get("replanner_shadow_report") if isinstance(state.get("replanner_shadow_report"), dict) else {}
    current_route = [str(item) for item in shadow.get("current_route") or route_plan.get("route", []) or []]
    candidate_route = [str(item) for item in shadow.get("suggested_route") or current_route]
    intent = str(route_plan.get("intent") or state.get("route") or "chat")
    risk_level = str(route_plan.get("risk_level") or shadow.get("safety_level") or "L0")
    blocked_reasons = [str(item) for item in shadow.get("blocked_reasons") or [] if item]
    warnings: list[str] = []

    if not shadow:
        _append_once(blocked_reasons, "missing_replanner_shadow_report")
        _append_once(warnings, "missing_replanner_shadow_report")
    if not bool(shadow.get("replan_recommended")):
        _append_once(blocked_reasons, "shadow_not_recommended")
    if intent not in _CANDIDATE_ALLOWED_INTENTS:
        _append_once(blocked_reasons, f"unsafe_intent:{intent}")
    if risk_level in {"L3", "L4"}:
        _append_once(blocked_reasons, f"risk_level:{risk_level}")
    for node in candidate_route:
        if node not in _CANDIDATE_ALLOWED_ROUTE_NODES:
            _append_once(blocked_reasons, f"candidate_node_not_allowed:{node}")
    for blocker in _blocked_reasons(state, intent, risk_level, current_route):
        _append_once(blocked_reasons, blocker)

    eligible = bool(shadow.get("replan_recommended")) and not blocked_reasons
    candidate_execution_plan = {}
    if candidate_route:
        candidate_route_plan = {
            **route_plan,
            "route": list(candidate_route),
            "intent": intent,
            "risk_level": risk_level,
            "reason": "replanner_candidate_shadow",
        }
        candidate_execution_plan = execution_plan_from_route_plan(candidate_route_plan, state)

    candidate = {
        "mode": "candidate_only",
        "candidate_id": f"candidate:{uuid4().hex[:12]}",
        "source": "replanner_shadow",
        "eligible": eligible,
        "current_route": current_route,
        "candidate_route": candidate_route,
        "candidate_execution_plan": candidate_execution_plan,
        "replan_reasons": [str(item) for item in shadow.get("replan_reasons") or [] if item],
        "blocked_reasons": blocked_reasons,
        "confidence": float(shadow.get("confidence") or 0.0),
    }
    return {
        "replanner_candidate_plan": candidate,
        "replanner_candidate_warnings": warnings,
    }


def build_replanner_control_decision(state: dict[str, Any], legacy_next_node: str) -> dict[str, Any]:
    """Return a limited replanner control decision for dispatch.

    The decision is intentionally conservative: it only chooses a low-risk
    candidate node, never mutates route state, and falls back to the supplied
    legacy dispatcher result on any doubt.
    """
    settings = get_settings()
    control_enabled = bool(getattr(settings, "agent_replanner_control_enabled", False))
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    candidate = state.get("replanner_candidate_plan") if isinstance(state.get("replanner_candidate_plan"), dict) else {}
    shadow = state.get("replanner_shadow_report") if isinstance(state.get("replanner_shadow_report"), dict) else {}
    supervisor_audit = state.get("supervisor_dispatch_audit") if isinstance(state.get("supervisor_dispatch_audit"), dict) else {}
    supervisor_policy = state.get("supervisor_shadow_policy") if isinstance(state.get("supervisor_shadow_policy"), dict) else {}

    intent = str(route_plan.get("intent") or state.get("route") or "chat")
    risk_level = str(route_plan.get("risk_level") or shadow.get("safety_level") or "L0")
    completed_nodes = {str(item) for item in state.get("completed_nodes", []) or []}
    candidate_route = [str(item) for item in candidate.get("candidate_route") or [] if item]
    selected_candidate_node = _next_candidate_node(candidate_route, completed_nodes)

    blockers: list[str] = []
    warnings: list[str] = []
    fallback_reason = ""
    selected_next_node = legacy_next_node
    control_applied = False

    if not control_enabled:
        fallback_reason = "control_disabled"
        _append_once(blockers, fallback_reason)
    if not candidate:
        _append_once(blockers, "missing_replanner_candidate_plan")
    elif not bool(candidate.get("eligible")):
        _append_once(blockers, "candidate_not_eligible")
    if intent not in _CANDIDATE_ALLOWED_INTENTS:
        _append_once(blockers, f"unsafe_intent:{intent}")
    if risk_level in {"L3", "L4"}:
        _append_once(blockers, f"risk_level:{risk_level}")
    for blocker in _blocked_reasons(state, intent, risk_level, _string_list(route_plan.get("route"))):
        _append_once(blockers, blocker)
    if supervisor_audit.get("status") == "mismatch":
        _append_once(blockers, "supervisor_dispatch_mismatch")
    for blocker in supervisor_policy.get("control_blockers") or []:
        _append_once(blockers, f"supervisor_control_blocker:{blocker}")
    if legacy_next_node == "__end__":
        _append_once(blockers, "graph_interrupt")
    if selected_candidate_node not in _CONTROL_ALLOWED_NODES:
        _append_once(blockers, f"candidate_node_not_allowed:{selected_candidate_node or 'missing'}")
    if legacy_next_node not in _CONTROL_ALLOWED_NODES:
        _append_once(blockers, f"legacy_node_not_allowed:{legacy_next_node}")
    if selected_candidate_node in completed_nodes and selected_candidate_node != "final_response":
        _append_once(blockers, f"retry_completed_agent:{selected_candidate_node}")
    for node in candidate_route:
        if node not in _CONTROL_ALLOWED_NODES:
            _append_once(blockers, f"candidate_node_not_allowed:{node}")

    if not fallback_reason and blockers:
        fallback_reason = blockers[0]
    if not blockers:
        selected_next_node = str(selected_candidate_node)
        control_applied = True
    else:
        _append_once(warnings, f"replanner_control_fallback:{fallback_reason}")

    decision = {
        "mode": "limited_control",
        "control_enabled": control_enabled,
        "control_applied": control_applied,
        "selected_next_node": selected_next_node,
        "legacy_next_node": legacy_next_node,
        "candidate_next_node": selected_candidate_node,
        "fallback_reason": "" if control_applied else fallback_reason,
        "blockers": blockers,
    }
    return {
        "replanner_control_decision": decision,
        "replanner_control_warnings": warnings,
    }


def update_replanner_shadow_metrics(state: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Return cumulative in-state metrics for replanner observations."""
    previous = state.get("replanner_shadow_metrics") if isinstance(state.get("replanner_shadow_metrics"), dict) else {}
    shadow = observation.get("replanner_shadow_report") if isinstance(observation.get("replanner_shadow_report"), dict) else {}
    candidate = observation.get("replanner_candidate_plan") if isinstance(observation.get("replanner_candidate_plan"), dict) else {}
    control = observation.get("replanner_control_decision") if isinstance(observation.get("replanner_control_decision"), dict) else {}
    warnings = (
        _string_list(observation.get("replanner_shadow_warnings"))
        + _string_list(observation.get("replanner_candidate_warnings"))
        + _string_list(observation.get("replanner_control_warnings"))
    )
    blockers = _string_list(candidate.get("blocked_reasons")) + _string_list(control.get("blockers"))

    metrics = {
        "enabled": True,
        "control_enabled": bool(control.get("control_enabled")),
        "shadow_observation_count": int(previous.get("shadow_observation_count") or 0) + 1,
        "candidate_eligible_count": int(previous.get("candidate_eligible_count") or 0),
        "candidate_blocked_count": int(previous.get("candidate_blocked_count") or 0),
        "control_applied_count": int(previous.get("control_applied_count") or 0),
        "control_fallback_count": int(previous.get("control_fallback_count") or 0),
        "warning_count": int(previous.get("warning_count") or 0) + len(warnings),
        "blocker_counts": dict(previous.get("blocker_counts") or {}),
        "latest_selected_node": control.get("selected_next_node"),
        "latest_legacy_node": control.get("legacy_next_node"),
        "latest_candidate_node": control.get("candidate_next_node") or _latest_candidate_node(candidate),
        "latest_candidate_eligible": bool(candidate.get("eligible")),
        "latest_replan_recommended": bool(shadow.get("replan_recommended")),
    }
    if candidate.get("eligible"):
        metrics["candidate_eligible_count"] += 1
    else:
        metrics["candidate_blocked_count"] += 1
    if control.get("control_applied"):
        metrics["control_applied_count"] += 1
    else:
        metrics["control_fallback_count"] += 1
    for blocker in blockers:
        metrics["blocker_counts"][blocker] = int(metrics["blocker_counts"].get(blocker) or 0) + 1
    return metrics


def _collect_supervisor_signals(
    state: dict[str, Any],
    trigger_sources: list[str],
    replan_reasons: list[str],
) -> None:
    decision = state.get("supervisor_decision") if isinstance(state.get("supervisor_decision"), dict) else {}
    if not decision:
        return
    reasons = [str(item) for item in decision.get("replan_reasons") or [] if item]
    if decision.get("should_replan_hint") or reasons:
        _append_once(trigger_sources, "supervisor_decision")
    for reason in reasons:
        _append_once(replan_reasons, reason)
    if decision.get("should_replan_hint") and not reasons:
        _append_once(replan_reasons, "supervisor_should_replan_hint")


def _collect_contract_signals(
    state: dict[str, Any],
    trigger_sources: list[str],
    replan_reasons: list[str],
) -> None:
    for warning in _string_list(state.get("runtime_contract_warnings")):
        if warning.startswith(_REPLAN_CONTRACT_PREFIXES):
            _append_once(trigger_sources, "runtime_contract")
            _append_once(replan_reasons, warning)


def _collect_agent_failure_signals(
    state: dict[str, Any],
    trigger_sources: list[str],
    replan_reasons: list[str],
) -> None:
    agent_results = state.get("agent_results") if isinstance(state.get("agent_results"), list) else []
    for result in agent_results:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        agent = str(result.get("agent") or "")
        status = str(result.get("status") or "")
        if metadata.get("source") == "prefetch" or agent.endswith("_prefetch"):
            continue
        if status in _FAILED_STATUSES:
            _append_once(trigger_sources, "agent_results")
            _append_once(replan_reasons, f"agent_failed:{agent}")


def _collect_shadow_warning_signals(
    state: dict[str, Any],
    trigger_sources: list[str],
    replan_reasons: list[str],
) -> None:
    warning_fields = (
        ("supervisor_dispatch_warnings", "supervisor_dispatch"),
        ("supervisor_control_warnings", "supervisor_control"),
    )
    for field, source in warning_fields:
        values = _string_list(state.get(field))
        if values:
            _append_once(trigger_sources, source)
            for value in values:
                _append_once(replan_reasons, value)


def _blocked_reasons(
    state: dict[str, Any],
    intent: str,
    risk_level: str,
    current_route: list[str],
) -> list[str]:
    blocked: list[str] = []
    if state.get("status") == "waiting_approval":
        _append_once(blocked, "waiting_approval")
    if state.get("approval_payload") or state.get("approval_required"):
        _append_once(blocked, "approval_pending")
    if state.get("pending_approval_id") or state.get("pending_tool_call_id") or state.get("pending_tool_name"):
        _append_once(blocked, "pending_tool_or_approval")
    if risk_level in {"L3", "L4"}:
        _append_once(blocked, f"risk_level:{risk_level}")
    if intent in _UNSAFE_INTENTS:
        _append_once(blocked, f"unsafe_intent:{intent}")
    for agent in current_route:
        if agent in _UNSAFE_ROUTE_AGENTS:
            _append_once(blocked, f"unsafe_agent:{agent}")
    return blocked


def _suggested_route(current_route: list[str], replan_recommended: bool) -> list[str]:
    if not replan_recommended:
        return list(current_route)
    if not current_route:
        return ["final_response"]
    if "final_response" not in current_route:
        return [*current_route, "final_response"]
    return list(current_route)


def _suggested_actions(
    replan_reasons: list[str],
    blocked_reasons: list[str],
    suggested_route: list[str],
) -> list[str]:
    if blocked_reasons:
        return ["continue_legacy_route", "keep_shadow_observation"]
    if not replan_reasons:
        return ["no_replan_needed"]
    actions = ["shadow_replan_candidate"]
    if suggested_route:
        actions.append("candidate_route_available")
    return actions


def _confidence(replan_reasons: list[str], blocked_reasons: list[str]) -> float:
    if not replan_reasons:
        return 0.0
    base = min(0.9, 0.45 + 0.1 * len(replan_reasons))
    if blocked_reasons:
        return min(base, 0.35)
    return base


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _next_candidate_node(candidate_route: list[str], completed_nodes: set[str]) -> str:
    for node in candidate_route:
        if node not in completed_nodes:
            return node
    if candidate_route:
        return "final_response"
    return ""


def _latest_candidate_node(candidate: dict[str, Any]) -> str | None:
    route = candidate.get("candidate_route") if isinstance(candidate.get("candidate_route"), list) else []
    if not route:
        return None
    return str(route[-1])


def _append_once(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
