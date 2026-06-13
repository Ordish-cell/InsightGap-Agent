"""Runtime dispatch helpers for LangGraph conditional edges."""

from __future__ import annotations

from src.web_app.agent.runtime.intent_schema import normalize_agent_name
from src.web_app.agent.runtime.state import AgentRuntimeState
from src.web_app.agent.runtime.replanner import (
    build_replanner_candidate_plan,
    build_replanner_control_decision,
    build_replanner_shadow_report,
    update_replanner_shadow_metrics,
)
from src.web_app.agent.runtime.supervisor import (
    audit_supervisor_dispatch,
    build_supervisor_control_decision,
    build_supervisor_readiness_report,
    update_supervisor_shadow_metrics,
)


END_SENTINEL = "__end__"


def after_permission(state: AgentRuntimeState) -> str:
    if state.get("route") in {"approval", "blocked"} or state.get("error"):
        return "done"
    return "continue"


def map_route_to_node(route_item: str) -> str:
    """Map a route_plan route item to a registered graph node name."""
    normalized = normalize_agent_name(route_item)
    mapping = {
        "context_builder": "context_builder",
        "skill_matcher": "skill_matcher",
        "research_agent": "research_agent",
        "rag_agent": "rag_agent",
        "artifact_agent": "artifact_agent",
        "tool_agent": "tool_agent",
        "memory_agent": "memory_agent",
        "skill_agent": "skill_agent",
        "evaluator": "evaluator",
        "final_response": "final_response",
    }
    return mapping.get(normalized, mapping.get(route_item, "final_response"))


def legacy_next_route_node(state: AgentRuntimeState) -> str:
    """Compute the next legacy route node without supervisor side effects."""
    if state.get("status") == "waiting_approval":
        return END_SENTINEL

    route_plan = state.get("route_plan") or {}
    route_list = list(route_plan.get("route", []))
    completed = list(state.get("completed_nodes", []))

    for node_name in route_list:
        if node_name not in completed:
            return map_route_to_node(node_name)

    return "final_response"


def record_supervisor_dispatch_audit(state: AgentRuntimeState, legacy_next_node: str) -> str:
    observation = audit_supervisor_dispatch(state, legacy_next_node)
    state["supervisor_dispatch_audit"] = observation["supervisor_dispatch_audit"]
    state["supervisor_dispatch_warnings"] = observation["supervisor_dispatch_warnings"]
    state["supervisor_shadow_policy"] = observation["supervisor_shadow_policy"]
    state["supervisor_policy_warnings"] = observation["supervisor_policy_warnings"]
    state["supervisor_shadow_metrics"] = update_supervisor_shadow_metrics(state, observation)
    readiness = build_supervisor_readiness_report(state)
    state["supervisor_readiness_report"] = readiness["supervisor_readiness_report"]
    state["supervisor_readiness_warnings"] = readiness["supervisor_readiness_warnings"]
    control = build_supervisor_control_decision(state, legacy_next_node)
    state["supervisor_control_decision"] = control["supervisor_control_decision"]
    state["supervisor_control_warnings"] = control["supervisor_control_warnings"]
    supervisor_next_node = state["supervisor_control_decision"].get("selected_next_node") or legacy_next_node

    replanner = build_replanner_shadow_report(state)
    state["replanner_shadow_report"] = replanner["replanner_shadow_report"]
    state["replanner_shadow_warnings"] = replanner["replanner_shadow_warnings"]
    candidate = build_replanner_candidate_plan(state)
    state["replanner_candidate_plan"] = candidate["replanner_candidate_plan"]
    state["replanner_candidate_warnings"] = candidate["replanner_candidate_warnings"]
    replanner_control = build_replanner_control_decision(state, supervisor_next_node)
    state["replanner_control_decision"] = replanner_control["replanner_control_decision"]
    state["replanner_control_warnings"] = replanner_control["replanner_control_warnings"]
    replanner_observation = {
        **replanner,
        **candidate,
        **replanner_control,
    }
    state["replanner_shadow_metrics"] = update_replanner_shadow_metrics(state, replanner_observation)
    return state["replanner_control_decision"].get("selected_next_node") or supervisor_next_node


def dispatch_next_route_node(state: AgentRuntimeState) -> str:
    """Return the next graph node while preserving supervisor observation side effects."""
    return record_supervisor_dispatch_audit(state, legacy_next_route_node(state))
