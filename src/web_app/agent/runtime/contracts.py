"""Runtime contract observation helpers.

The contract report is intentionally read-only. It summarizes whether the
standardized runtime signals line up, but never gates dispatch or final output.
"""

from __future__ import annotations

from typing import Any

_FORMAL_AGENT_NODES = {
    "research_agent",
    "rag_agent",
    "artifact_agent",
    "tool_agent",
    "memory_agent",
    "skill_agent",
    "evaluator",
    "final_response",
}
_REPLANNER_CONTROL_ALLOWLIST = {"rag_agent", "final_response"}


def build_runtime_contract_report(state: dict[str, Any]) -> dict[str, Any]:
    from src.web_app.agent.runtime.graph_manifest import build_runtime_graph_manifest

    manifest = build_runtime_graph_manifest()
    warnings: list[str] = []
    node_results = [item for item in state.get("node_results", []) or [] if isinstance(item, dict)]
    agent_results = [item for item in state.get("agent_results", []) or [] if isinstance(item, dict)]
    completed_nodes = [str(item) for item in state.get("completed_nodes", []) or []]
    route_plan = state.get("route_plan") or {}
    route_nodes = [str(item) for item in route_plan.get("route", []) or []]
    node_result_names = [str(item.get("node") or "") for item in node_results if item.get("node")]

    expected_nodes = _expected_node_results(route_nodes, state)
    missing_node_results = [
        node for node in expected_nodes
        if node not in node_result_names and not _is_legal_waiting_approval_exception(state, node)
    ]
    for node in missing_node_results:
        warnings.append(f"missing_node_result:{node}")

    completion = _build_completion_consistency(
        state=state,
        route_nodes=route_nodes,
        completed_nodes=completed_nodes,
        node_results=node_results,
        warnings=warnings,
    )
    agent_consistency = _build_agent_result_consistency(
        agent_results=agent_results,
        node_results=node_results,
        warnings=warnings,
    )
    manifest_consistency = _build_manifest_consistency(
        manifest=manifest,
        node_results=node_results,
        warnings=warnings,
    )
    replanner_consistency = _build_replanner_consistency(
        state=state,
        warnings=warnings,
    )

    report = {
        "mode": "observe_only",
        "node_result_coverage": {
            "expected_nodes": expected_nodes,
            "observed_nodes": node_result_names,
            "missing_nodes": missing_node_results,
            "coverage_ok": not missing_node_results,
        },
        "completion_consistency": completion,
        "agent_result_consistency": agent_consistency,
        "manifest_consistency": manifest_consistency,
        "replanner_consistency": replanner_consistency,
        "route": route_nodes,
        "status": state.get("status"),
    }
    return {
        "runtime_contract_report": report,
        "runtime_contract_warnings": warnings,
    }


def _expected_node_results(route_nodes: list[str], state: dict[str, Any]) -> list[str]:
    expected: list[str] = []
    for node in route_nodes:
        if node and node not in expected:
            expected.append(node)
    if (state.get("final_payload") or state.get("final_answer") is not None) and "final_response" not in expected:
        expected.append("final_response")
    return expected


def _build_completion_consistency(
    *,
    state: dict[str, Any],
    route_nodes: list[str],
    completed_nodes: list[str],
    node_results: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    node_result_names = {str(item.get("node") or "") for item in node_results if item.get("node")}
    completed_without_node_result = [
        node for node in completed_nodes
        if node in _FORMAL_AGENT_NODES and node not in node_result_names
    ]
    for node in completed_without_node_result:
        warnings.append(f"completed_without_node_result:{node}")

    route_nodes_not_completed: list[str] = []
    if state.get("status") != "waiting_approval":
        for node in route_nodes:
            if node in {"evaluator", "final_response"}:
                continue
            if node not in completed_nodes:
                route_nodes_not_completed.append(node)
                warnings.append(f"route_node_not_completed:{node}")
    elif "tool_agent" in route_nodes and "tool_agent" in node_result_names:
        # Tool approval pause is a valid interrupt: node_result exists, but the
        # tool node must not be marked completed yet.
        pass

    return {
        "completed_nodes": completed_nodes,
        "completed_without_node_result": completed_without_node_result,
        "route_nodes_not_completed": route_nodes_not_completed,
        "waiting_approval_tool_exception": bool(
            state.get("status") == "waiting_approval"
            and "tool_agent" in route_nodes
            and "tool_agent" in node_result_names
            and "tool_agent" not in completed_nodes
        ),
        "ok": not completed_without_node_result and not route_nodes_not_completed,
    }


def _build_agent_result_consistency(
    *,
    agent_results: list[dict[str, Any]],
    node_results: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    latest_by_agent: dict[str, dict[str, Any]] = {}
    for result in agent_results:
        agent = str(result.get("agent") or "")
        if agent in _FORMAL_AGENT_NODES:
            latest_by_agent[agent] = result

    latest_node_by_name: dict[str, dict[str, Any]] = {}
    for result in node_results:
        node = str(result.get("node") or "")
        if node:
            latest_node_by_name[node] = result

    missing_node_results: list[str] = []
    status_mismatches: list[dict[str, str]] = []
    for agent, agent_result in latest_by_agent.items():
        node_result = latest_node_by_name.get(agent)
        if not node_result:
            missing_node_results.append(agent)
            warnings.append(f"agent_result_without_node_result:{agent}")
            continue
        agent_status = str(agent_result.get("status") or "ok")
        node_status = str(node_result.get("status") or "ok")
        if agent_status != node_status:
            status_mismatches.append({
                "agent": agent,
                "agent_status": agent_status,
                "node_status": node_status,
            })
            warnings.append(f"agent_node_status_mismatch:{agent}")

    return {
        "agents": sorted(latest_by_agent),
        "missing_node_results": missing_node_results,
        "status_mismatches": status_mismatches,
        "ok": not missing_node_results and not status_mismatches,
    }


def _build_manifest_consistency(
    *,
    manifest: dict[str, Any],
    node_results: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    writes_by_node = {
        str(node.get("name")): set(node.get("writes") or [])
        for node in manifest.get("nodes", []) or []
        if isinstance(node, dict)
    }
    unknown_update_fields: list[dict[str, Any]] = []
    for result in node_results:
        node = str(result.get("node") or "")
        allowed = writes_by_node.get(node)
        if not node or allowed is None:
            continue
        updates = ((result.get("delta") or {}).get("updates") or {})
        if not isinstance(updates, dict):
            continue
        extras = sorted(key for key in updates if key not in allowed)
        if extras:
            unknown_update_fields.append({"node": node, "fields": extras})
            warnings.append(f"manifest_write_contract_extra:{node}")

    return {
        "unknown_update_fields": unknown_update_fields,
        "ok": not unknown_update_fields,
    }


def _build_replanner_consistency(
    *,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    shadow = state.get("replanner_shadow_report") if isinstance(state.get("replanner_shadow_report"), dict) else {}
    candidate = state.get("replanner_candidate_plan") if isinstance(state.get("replanner_candidate_plan"), dict) else {}
    control = state.get("replanner_control_decision") if isinstance(state.get("replanner_control_decision"), dict) else {}
    has_replanner_payload = bool(shadow or candidate or control)
    if not has_replanner_payload:
        return {
            "shadow_report_present": False,
            "candidate_plan_present": False,
            "candidate_route_from_shadow": False,
            "control_selected_node_allowed": False,
            "control_applied_low_risk": False,
            "ok": True,
        }

    candidate_route = [str(item) for item in candidate.get("candidate_route") or []]
    shadow_suggested_route = [str(item) for item in shadow.get("suggested_route") or []]
    selected_node = str(control.get("selected_next_node") or "")
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    risk_level = str(route_plan.get("risk_level") or shadow.get("safety_level") or "L0")
    intent = str(route_plan.get("intent") or state.get("route") or "chat")

    shadow_present = bool(shadow)
    candidate_present = bool(candidate)
    candidate_route_from_shadow = candidate_route == shadow_suggested_route
    selected_allowed = selected_node in _REPLANNER_CONTROL_ALLOWLIST
    control_applied = bool(control.get("control_applied"))
    control_applied_low_risk = (
        not control_applied
        or (
            intent in {"chat", "rag"}
            and risk_level not in {"L3", "L4"}
            and selected_allowed
        )
    )

    if not shadow_present:
        warnings.append("replanner_missing_shadow_report")
    if not candidate_present:
        warnings.append("replanner_missing_candidate_plan")
    if candidate_present and shadow_present and not candidate_route_from_shadow:
        warnings.append("replanner_candidate_route_not_from_shadow")
    if control and not selected_allowed:
        warnings.append(f"replanner_control_selected_node_not_allowed:{selected_node or 'missing'}")
    if control_applied and not control_applied_low_risk:
        warnings.append("replanner_control_applied_non_low_risk")

    return {
        "shadow_report_present": shadow_present,
        "candidate_plan_present": candidate_present,
        "candidate_route_from_shadow": candidate_route_from_shadow,
        "control_selected_node_allowed": selected_allowed,
        "control_applied_low_risk": control_applied_low_risk,
        "ok": (
            shadow_present
            and candidate_present
            and candidate_route_from_shadow
            and (not control or selected_allowed)
            and control_applied_low_risk
        ),
    }


def _is_legal_waiting_approval_exception(state: dict[str, Any], node: str) -> bool:
    return bool(state.get("status") == "waiting_approval" and node == "tool_agent")
