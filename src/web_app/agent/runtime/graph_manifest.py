"""Static runtime graph manifest and debug rendering."""

from __future__ import annotations

from typing import Any

from src.web_app.agent.runtime.dispatch import END_SENTINEL
from src.web_app.agent.runtime.graph_registry import (
    AGENT_NODE_NAMES,
    FALLBACK_NODE_NAMES,
    GRAPH_NODE_NAMES,
    ROUTE_DESTINATION_NODE_NAMES,
    RUNTIME_NODE_SPECS,
)
from src.web_app.core.config import settings


MAIN_CHAIN_EDGES: tuple[dict[str, str], ...] = (
    {"from": "permission_guard", "to": "home_intent_react", "condition": "continue"},
    {"from": "permission_guard", "to": "final_response", "condition": "done"},
    {"from": "home_intent_react", "to": "planner"},
    {"from": "planner", "to": "parallel_prefetch"},
    {"from": "parallel_prefetch", "to": "parallel_read_stage"},
    {"from": "parallel_read_stage", "to": "supervisor_observer"},
    {"from": "supervisor_observer", "to": "llm_supervisor_route"},
    {"from": "llm_supervisor_route", "to": "route_dispatch", "condition": "dispatch_next_route_node"},
    {"from": "route_dispatch", "to": "research_agent", "condition": "research_agent"},
    {"from": "route_dispatch", "to": "rag_agent", "condition": "rag_agent"},
    {"from": "route_dispatch", "to": "artifact_agent", "condition": "artifact_agent"},
    {"from": "route_dispatch", "to": "tool_agent", "condition": "tool_agent"},
    {"from": "route_dispatch", "to": "memory_agent", "condition": "memory_agent"},
    {"from": "route_dispatch", "to": "skill_agent", "condition": "skill_agent"},
    {"from": "route_dispatch", "to": "evaluator", "condition": "evaluator"},
    {"from": "route_dispatch", "to": "final_response", "condition": "final_response"},
    {"from": "route_dispatch", "to": END_SENTINEL, "condition": "waiting_approval"},
    {"from": "evaluator", "to": "final_response"},
    {"from": "final_response", "to": END_SENTINEL},
)


def build_runtime_graph_manifest() -> dict[str, Any]:
    return {
        "nodes": [_node_spec_to_dict(spec) for spec in RUNTIME_NODE_SPECS],
        "edges": [dict(edge) for edge in MAIN_CHAIN_EDGES],
        "route_destinations": list(ROUTE_DESTINATION_NODE_NAMES),
        "fallback_sequence": list(FALLBACK_NODE_NAMES),
        "graph_node_names": list(GRAPH_NODE_NAMES),
        "agent_node_names": list(AGENT_NODE_NAMES),
        "checkpointer_default": bool(getattr(settings, "agent_langgraph_checkpointer_enabled", False)),
        "supervisor_control_default": bool(getattr(settings, "agent_supervisor_control_enabled", False)),
        "llm_supervisor_capabilities": {
            "route_plan_override": True,
            "enabled_default": bool(getattr(settings, "agent_llm_supervisor_enabled", False)),
            "mode_default": str(getattr(settings, "agent_llm_supervisor_mode", "shadow")),
            "dispatch_reads_route_plan_only": True,
        },
        "replanner_capabilities": {
            "shadow_observation": True,
            "candidate_plan": True,
            "limited_control": True,
            "control_default": bool(getattr(settings, "agent_replanner_control_enabled", False)),
            "control_allowlist": ["rag_agent", "final_response"],
            "dispatch_observer": True,
            "graph_node": False,
        },
    }


def render_runtime_graph_mermaid() -> str:
    return "\n".join([
        "flowchart TD",
        '    permission_guard["permission_guard"] -->|continue| home_intent_react["home_intent_react"]',
        '    permission_guard -->|done| final_response["final_response"]',
        '    home_intent_react --> planner["planner"]',
        '    planner --> parallel_prefetch["parallel_prefetch"]',
        '    parallel_prefetch --> parallel_read_stage["parallel_read_stage"]',
        '    parallel_read_stage --> supervisor_observer["supervisor_observer"]',
        '    supervisor_observer --> llm_supervisor_route["llm_supervisor_route"]',
        '    llm_supervisor_route --> route_dispatch{"route dispatch"}',
        '    route_dispatch -. "replanner dispatch observer" .-> route_dispatch',
        '    route_dispatch --> research_agent["research_agent"]',
        '    route_dispatch --> rag_agent["rag_agent"]',
        '    route_dispatch --> artifact_agent["artifact_agent"]',
        '    route_dispatch --> tool_agent["tool_agent"]',
        '    route_dispatch --> memory_agent["memory_agent"]',
        '    route_dispatch --> skill_agent["skill_agent"]',
        '    research_agent --> route_dispatch',
        '    rag_agent --> route_dispatch',
        '    artifact_agent --> route_dispatch',
        '    tool_agent --> route_dispatch',
        '    memory_agent --> route_dispatch',
        '    skill_agent --> route_dispatch',
        '    route_dispatch --> evaluator["evaluator"]',
        '    evaluator --> final_response',
        '    route_dispatch -->|waiting approval| END["END"]',
        '    final_response --> END',
    ])


def _node_spec_to_dict(spec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "attr_name": spec.attr_name,
        "kind": spec.kind,
        "is_route_destination": spec.is_route_destination,
        "runnable_enabled": spec.runnable_enabled,
        "reads": list(spec.reads),
        "writes": list(spec.writes),
        "side_effect_level": spec.side_effect_level,
        "description": spec.description,
    }
