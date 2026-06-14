from __future__ import annotations

from typing import Any

from src.web_app.agent.runtime.schemas import SupervisorDecision
from src.web_app.core.config import get_settings


_FORMAL_AGENTS = {
    "research_agent",
    "rag_agent",
    "artifact_agent",
    "tool_agent",
    "memory_agent",
    "skill_agent",
    "evaluator",
    "final_response",
}

_UNSAFE_CONTROL_AGENTS = {
    "tool_agent",
    "artifact_agent",
    "memory_agent",
    "skill_agent",
    "research_agent",
}
_UNSAFE_CONTROL_INTENTS = {
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
_LIMITED_CONTROL_NODES = {"rag_agent", "final_response"}


def observe_supervisor_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build an observe-only supervisor snapshot from existing runtime state.

    This function intentionally performs no I/O and does not mutate state. The
    RuntimeNodes facade is responsible for writing the returned supervisor_*
    fields back into the graph state.
    """
    if not _settings_flag("agent_supervisor_enabled", True):
        return {
            "supervisor_decision": {},
            "supervisor_warnings": ["supervisor_disabled"],
            "supervisor_trace": [{"event": "supervisor_disabled"}],
        }

    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    execution_plan = state.get("execution_plan") if isinstance(state.get("execution_plan"), dict) else {}
    current_route = _route_from_plan(route_plan, execution_plan)
    completed_nodes = _string_list(state.get("completed_nodes"))
    completed_set = set(completed_nodes)
    pending_nodes = [node for node in current_route if node not in completed_set]
    next_expected_node = pending_nodes[0] if pending_nodes else None

    prefetch_results = state.get("prefetch_results") if isinstance(state.get("prefetch_results"), dict) else {}
    parallel_read_results = state.get("parallel_read_results") if isinstance(state.get("parallel_read_results"), dict) else {}
    rag_prepare = parallel_read_results.get("rag_prepare") if isinstance(parallel_read_results.get("rag_prepare"), dict) else {}
    rag_prepare_evidence = list(rag_prepare.get("evidence") or []) if isinstance(rag_prepare, dict) else []

    failed_agents, replan_reasons, warnings = _inspect_agent_results(state.get("agent_results"))
    waiting_approval = state.get("status") == "waiting_approval"
    if waiting_approval:
        _append_once(warnings, "approval_pending")

    trace = [
        {
            "event": "route_observed",
            "route_length": len(current_route),
            "completed_count": len(completed_nodes),
            "pending_count": len(pending_nodes),
        },
        {
            "event": "context_observed",
            "has_prefetch_context": _has_context(prefetch_results),
            "has_parallel_read_context": _has_context(parallel_read_results),
            "rag_prepare_evidence_count": len(rag_prepare_evidence),
        },
    ]
    if failed_agents:
        trace.append({"event": "agent_failures_observed", "agents": failed_agents})
    if waiting_approval:
        trace.append({"event": "approval_wait_observed"})

    decision = SupervisorDecision(
        current_intent=str(route_plan.get("intent") or execution_plan.get("intent") or "") or None,
        current_route=current_route,
        next_expected_node=next_expected_node,
        observed_completed_nodes=completed_nodes,
        observed_pending_nodes=pending_nodes,
        has_prefetch_context=_has_context(prefetch_results),
        has_parallel_read_context=_has_context(parallel_read_results),
        has_rag_prepare_evidence=bool(rag_prepare_evidence),
        rag_prepare_evidence_count=len(rag_prepare_evidence),
        failed_agents=failed_agents,
        waiting_approval=waiting_approval,
        should_replan_hint=bool(failed_agents or replan_reasons),
        replan_reasons=replan_reasons,
        warnings=warnings,
        trace=trace,
    ).model_dump()
    return {
        "supervisor_decision": decision,
        "supervisor_warnings": list(warnings),
        "supervisor_trace": list(trace),
    }


def audit_supervisor_dispatch(state: dict[str, Any], legacy_next_node: str) -> dict[str, Any]:
    """Compare observe-only supervisor expectation with legacy dispatcher output.

    The audit never decides routing. The caller must keep returning the legacy
    next node regardless of the audit result.
    """
    if not _settings_flag("agent_supervisor_enabled", True):
        audit = {
            "status": "skipped",
            "reason": "supervisor_disabled",
            "expected_next_node": None,
            "legacy_next_node": legacy_next_node,
            "matched": None,
        }
        policy = build_supervisor_shadow_policy(state, legacy_next_node)
        return {
            "supervisor_dispatch_audit": audit,
            "supervisor_dispatch_warnings": [],
            "supervisor_shadow_policy": policy,
            "supervisor_policy_warnings": list(policy.get("policy_warnings") or []),
        }

    decision = state.get("supervisor_decision") if isinstance(state.get("supervisor_decision"), dict) else {}
    warnings: list[str] = []
    if not decision:
        audit = {
            "status": "skipped",
            "reason": "missing_supervisor_decision",
            "expected_next_node": None,
            "legacy_next_node": legacy_next_node,
            "matched": None,
        }
        policy = build_supervisor_shadow_policy(state, legacy_next_node)
        return {
            "supervisor_dispatch_audit": audit,
            "supervisor_dispatch_warnings": warnings,
            "supervisor_shadow_policy": policy,
            "supervisor_policy_warnings": list(policy.get("policy_warnings") or []),
        }

    expected_next_node = decision.get("next_expected_node")
    if state.get("status") == "waiting_approval":
        audit = {
            "status": "skipped",
            "reason": "waiting_approval",
            "expected_next_node": expected_next_node,
            "legacy_next_node": legacy_next_node,
            "matched": None,
        }
        policy = build_supervisor_shadow_policy(state, legacy_next_node)
        return {
            "supervisor_dispatch_audit": audit,
            "supervisor_dispatch_warnings": warnings,
            "supervisor_shadow_policy": policy,
            "supervisor_policy_warnings": list(policy.get("policy_warnings") or []),
        }

    matched = expected_next_node == legacy_next_node or (
        expected_next_node is None and legacy_next_node == "final_response"
    )
    status = "ok" if matched else "mismatch"
    if not matched:
        _append_once(warnings, "supervisor_dispatch_mismatch")
    audit = {
        "status": status,
        "expected_next_node": expected_next_node,
        "legacy_next_node": legacy_next_node,
        "matched": matched,
        "observed_pending_nodes": list(decision.get("observed_pending_nodes") or []),
        "observed_completed_nodes": list(decision.get("observed_completed_nodes") or []),
    }
    policy = build_supervisor_shadow_policy(state, legacy_next_node)
    return {
        "supervisor_dispatch_audit": audit,
        "supervisor_dispatch_warnings": warnings,
        "supervisor_shadow_policy": policy,
        "supervisor_policy_warnings": list(policy.get("policy_warnings") or []),
    }


def build_supervisor_shadow_policy(state: dict[str, Any], legacy_next_node: str) -> dict[str, Any]:
    """Build a non-controlling supervisor dispatch policy recommendation."""
    if not _settings_flag("agent_supervisor_enabled", True):
        return _disabled_policy(legacy_next_node, "supervisor_disabled")
    if not _settings_flag("agent_supervisor_shadow_policy_enabled", True):
        return _disabled_policy(legacy_next_node, "shadow_policy_disabled")

    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    decision = state.get("supervisor_decision") if isinstance(state.get("supervisor_decision"), dict) else {}
    current_route = [str(item) for item in route_plan.get("route", [])] if isinstance(route_plan.get("route"), list) else []
    intent = str(route_plan.get("intent") or decision.get("current_intent") or state.get("route") or "")
    risk_level = str(route_plan.get("risk_level") or "L0")
    recommended_next_node = decision.get("next_expected_node") if decision else None
    if recommended_next_node is None and legacy_next_node == "final_response":
        recommended_next_node = "final_response"

    blockers: list[str] = []
    warnings: list[str] = []
    if state.get("status") == "waiting_approval":
        _append_once(blockers, "waiting_approval")
    if state.get("approval_payload") or state.get("approval_required"):
        _append_once(blockers, "approval_pending")
    if state.get("pending_approval_id") or state.get("pending_tool_call_id") or state.get("pending_tool_name"):
        _append_once(blockers, "pending_tool_or_approval")
    if intent in _UNSAFE_CONTROL_INTENTS:
        _append_once(blockers, f"unsafe_intent:{intent}")
    for agent in current_route:
        if agent in _UNSAFE_CONTROL_AGENTS:
            _append_once(blockers, f"unsafe_agent:{agent}")
    if risk_level in {"L3", "L4"}:
        _append_once(blockers, f"risk_level:{risk_level}")
    if legacy_next_node == "__end__":
        _append_once(blockers, "graph_interrupt")

    if blockers:
        _append_once(warnings, "supervisor_control_blocked")

    return {
        "mode": "shadow_only",
        "legacy_next_node": legacy_next_node,
        "recommended_next_node": recommended_next_node,
        "control_eligible": not blockers,
        "control_blockers": blockers,
        "policy_warnings": warnings,
        "control_enabled": bool(_settings_flag("agent_supervisor_control_enabled", False)),
    }


def update_supervisor_shadow_metrics(state: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Return cumulative in-state metrics for supervisor shadow observations."""
    if not _settings_flag("agent_supervisor_shadow_metrics_enabled", True):
        return {"enabled": False}

    previous = state.get("supervisor_shadow_metrics") if isinstance(state.get("supervisor_shadow_metrics"), dict) else {}
    audit = observation.get("supervisor_dispatch_audit") if isinstance(observation.get("supervisor_dispatch_audit"), dict) else {}
    policy = observation.get("supervisor_shadow_policy") if isinstance(observation.get("supervisor_shadow_policy"), dict) else {}
    warnings = list(observation.get("supervisor_dispatch_warnings") or []) + list(observation.get("supervisor_policy_warnings") or [])
    blockers = list(policy.get("control_blockers") or [])
    metrics = {
        "enabled": True,
        "supervisor_enabled": bool(_settings_flag("agent_supervisor_enabled", True)),
        "shadow_policy_enabled": bool(_settings_flag("agent_supervisor_shadow_policy_enabled", True)),
        "control_enabled": bool(_settings_flag("agent_supervisor_control_enabled", False)),
        "dispatch_audit_count": int(previous.get("dispatch_audit_count") or 0) + 1,
        "dispatch_mismatch_count": int(previous.get("dispatch_mismatch_count") or 0),
        "dispatch_skipped_count": int(previous.get("dispatch_skipped_count") or 0),
        "control_eligible_count": int(previous.get("control_eligible_count") or 0),
        "control_blocked_count": int(previous.get("control_blocked_count") or 0),
        "warning_count": int(previous.get("warning_count") or 0) + len(warnings),
        "blocker_counts": dict(previous.get("blocker_counts") or {}),
        "latest_legacy_next_node": audit.get("legacy_next_node") or policy.get("legacy_next_node"),
        "latest_recommended_next_node": policy.get("recommended_next_node"),
        "latest_audit_status": audit.get("status"),
        "latest_control_eligible": bool(policy.get("control_eligible")),
    }
    if audit.get("status") == "mismatch":
        metrics["dispatch_mismatch_count"] += 1
    if audit.get("status") == "skipped":
        metrics["dispatch_skipped_count"] += 1
    if policy.get("control_eligible"):
        metrics["control_eligible_count"] += 1
    else:
        metrics["control_blocked_count"] += 1
    for blocker in blockers:
        metrics["blocker_counts"][blocker] = int(metrics["blocker_counts"].get(blocker) or 0) + 1
    return metrics


def build_supervisor_readiness_report(state: dict[str, Any]) -> dict[str, Any]:
    """Build a read-only readiness gate from existing supervisor shadow signals.

    P4E is still observation-only: this function does not decide routing, does
    not call external services, and does not mutate the input state.
    """
    metrics = state.get("supervisor_shadow_metrics") if isinstance(state.get("supervisor_shadow_metrics"), dict) else {}
    policy = state.get("supervisor_shadow_policy") if isinstance(state.get("supervisor_shadow_policy"), dict) else {}
    audit = state.get("supervisor_dispatch_audit") if isinstance(state.get("supervisor_dispatch_audit"), dict) else {}

    blockers: list[str] = []
    warnings: list[str] = []

    if not metrics:
        _append_once(blockers, "missing_shadow_metrics")
    if not policy:
        _append_once(blockers, "missing_shadow_policy")
    if not audit:
        _append_once(blockers, "missing_dispatch_audit")

    if metrics and metrics.get("enabled") is False:
        _append_once(blockers, "metrics_disabled")
    if policy and policy.get("mode") == "disabled":
        _append_once(blockers, "policy_disabled")
    if policy and not bool(policy.get("control_eligible")):
        _append_once(blockers, "policy_not_control_eligible")
    if audit.get("status") == "mismatch":
        _append_once(blockers, "dispatch_mismatch")
    if audit.get("status") == "skipped":
        reason = str(audit.get("reason") or "unknown")
        _append_once(blockers, f"dispatch_skipped:{reason}")

    for blocker in policy.get("control_blockers") or []:
        _append_once(blockers, str(blocker))
    for warning in (policy.get("policy_warnings") or []):
        _append_once(warnings, str(warning))

    mismatch_count = int(metrics.get("dispatch_mismatch_count") or 0) if metrics else 0
    skipped_count = int(metrics.get("dispatch_skipped_count") or 0) if metrics else 0
    if mismatch_count:
        _append_once(blockers, "metrics_dispatch_mismatch")
    if skipped_count:
        _append_once(blockers, "metrics_dispatch_skipped")

    ready_for_control = (
        not blockers
        and bool(metrics.get("enabled"))
        and audit.get("status") == "ok"
        and policy.get("mode") == "shadow_only"
        and bool(policy.get("control_eligible"))
    )
    if ready_for_control:
        readiness_level = "eligible_candidate"
        recommended_next_phase = "p5a_candidate"
    elif blockers:
        readiness_level = "blocked"
        recommended_next_phase = "continue_shadow"
    else:
        readiness_level = "shadow_stable"
        recommended_next_phase = "continue_shadow"

    if bool(policy.get("control_enabled")):
        _append_once(warnings, "control_flag_observed_but_not_applied")

    metrics_summary = {
        "enabled": bool(metrics.get("enabled")),
        "supervisor_enabled": bool(metrics.get("supervisor_enabled")),
        "shadow_policy_enabled": bool(metrics.get("shadow_policy_enabled")),
        "control_enabled": bool(metrics.get("control_enabled") or policy.get("control_enabled")),
        "dispatch_audit_count": int(metrics.get("dispatch_audit_count") or 0),
        "dispatch_mismatch_count": mismatch_count,
        "dispatch_skipped_count": skipped_count,
        "control_eligible_count": int(metrics.get("control_eligible_count") or 0),
        "control_blocked_count": int(metrics.get("control_blocked_count") or 0),
        "latest_audit_status": audit.get("status") or metrics.get("latest_audit_status"),
        "legacy_next_node": audit.get("legacy_next_node") or policy.get("legacy_next_node") or metrics.get("latest_legacy_next_node"),
        "recommended_next_node": policy.get("recommended_next_node") or metrics.get("latest_recommended_next_node"),
    }
    report = {
        "mode": "readiness_only",
        "ready_for_control": ready_for_control,
        "readiness_level": readiness_level,
        "metrics_summary": metrics_summary,
        "blockers": blockers,
        "recommended_next_phase": recommended_next_phase,
    }
    return {
        "supervisor_readiness_report": report,
        "supervisor_readiness_warnings": warnings,
    }


def build_supervisor_control_decision(state: dict[str, Any], legacy_next_node: str) -> dict[str, Any]:
    """Return a limited supervisor control decision for dispatch.

    P5A can only select a low-risk node that already passed shadow readiness.
    It never mutates state and falls back to the legacy dispatcher on any doubt.
    """
    readiness = state.get("supervisor_readiness_report") if isinstance(state.get("supervisor_readiness_report"), dict) else {}
    policy = state.get("supervisor_shadow_policy") if isinstance(state.get("supervisor_shadow_policy"), dict) else {}
    audit = state.get("supervisor_dispatch_audit") if isinstance(state.get("supervisor_dispatch_audit"), dict) else {}
    recommended_next_node = policy.get("recommended_next_node")

    blockers: list[str] = []
    warnings: list[str] = []
    fallback_reason = ""
    selected_next_node = legacy_next_node
    control_applied = False

    if not bool(policy.get("control_enabled")):
        fallback_reason = "control_disabled"
        _append_once(blockers, fallback_reason)
    if not readiness:
        _append_once(blockers, "missing_readiness_report")
    if not policy:
        _append_once(blockers, "missing_shadow_policy")
    if not audit:
        _append_once(blockers, "missing_dispatch_audit")
    if readiness and not bool(readiness.get("ready_for_control")):
        _append_once(blockers, "readiness_not_eligible")
    if readiness and readiness.get("readiness_level") != "eligible_candidate":
        _append_once(blockers, f"readiness_level:{readiness.get('readiness_level') or 'unknown'}")
    if audit.get("status") != "ok":
        _append_once(blockers, f"audit_status:{audit.get('status') or 'missing'}")
    if legacy_next_node not in _LIMITED_CONTROL_NODES:
        _append_once(blockers, f"legacy_node_not_allowed:{legacy_next_node}")
    if recommended_next_node not in _LIMITED_CONTROL_NODES:
        _append_once(blockers, f"recommended_node_not_allowed:{recommended_next_node or 'missing'}")
    for blocker in readiness.get("blockers") or []:
        _append_once(blockers, str(blocker))

    if not fallback_reason and blockers:
        fallback_reason = blockers[0]
    if not blockers:
        selected_next_node = str(recommended_next_node)
        control_applied = True
    else:
        _append_once(warnings, f"supervisor_control_fallback:{fallback_reason}")

    decision = {
        "mode": "limited_control",
        "control_applied": control_applied,
        "selected_next_node": selected_next_node,
        "legacy_next_node": legacy_next_node,
        "recommended_next_node": recommended_next_node,
        "fallback_reason": "" if control_applied else fallback_reason,
        "blockers": blockers,
    }
    return {
        "supervisor_control_decision": decision,
        "supervisor_control_warnings": warnings,
    }


def _disabled_policy(legacy_next_node: str, reason: str) -> dict[str, Any]:
    return {
        "mode": "disabled",
        "legacy_next_node": legacy_next_node,
        "recommended_next_node": None,
        "control_eligible": False,
        "control_blockers": [reason],
        "policy_warnings": [reason],
        "control_enabled": False,
    }


def _route_from_plan(route_plan: dict[str, Any], execution_plan: dict[str, Any]) -> list[str]:
    route = route_plan.get("route")
    if isinstance(route, list):
        return [str(node) for node in route]
    tasks = execution_plan.get("tasks") if isinstance(execution_plan, dict) else []
    if isinstance(tasks, list):
        return [str(task.get("agent")) for task in tasks if isinstance(task, dict) and task.get("agent")]
    return []


def _inspect_agent_results(agent_results: Any) -> tuple[list[str], list[str], list[str]]:
    failed_agents: list[str] = []
    replan_reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(agent_results, list):
        return failed_agents, replan_reasons, warnings
    for item in agent_results:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent") or "")
        if not _is_formal_agent_result(agent, item):
            continue
        status = str(item.get("status") or "")
        if status in {"failed", "timeout", "denied"}:
            _append_once(failed_agents, agent)
            _append_once(warnings, f"agent_failed:{agent}")
            _append_once(replan_reasons, f"agent_failed:{agent}")
    return failed_agents, replan_reasons, warnings


def _is_formal_agent_result(agent: str, result: dict[str, Any]) -> bool:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if metadata.get("source") == "prefetch":
        return False
    if agent.endswith("_prefetch"):
        return False
    return agent in _FORMAL_AGENTS


def _has_context(value: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for item in value.values():
        if item:
            return True
    return False


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _append_once(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _settings_flag(name: str, default: bool) -> bool:
    return bool(getattr(get_settings(), name, default))
