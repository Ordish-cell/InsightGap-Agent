"""LangGraph builder for the agent runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.web_app.agent.runtime.dispatch import END_SENTINEL
from src.web_app.agent.runtime.graph_registry import (
    AGENT_NODE_NAMES,
    ROUTE_DESTINATION_NODE_NAMES,
    build_runtime_node_registry,
)
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state import AgentRuntimeState


def build_agent_runtime_graph(
    nodes: RuntimeNodes,
    *,
    after_permission: Callable[[AgentRuntimeState], str],
    dispatch_next_route_node: Callable[[AgentRuntimeState], str],
    dispatch_after_evaluator: Callable[[AgentRuntimeState], str],
    checkpointer: Any | None = None,
) -> Any | None:
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return None

    workflow = StateGraph(AgentRuntimeState)

    for name, node_callable in build_runtime_node_registry(nodes).items():
        workflow.add_node(name, node_callable)

    route_dests = {name: name for name in ROUTE_DESTINATION_NODE_NAMES}
    route_dests[END_SENTINEL] = END

    workflow.set_entry_point("permission_guard")
    workflow.add_conditional_edges(
        "permission_guard",
        after_permission,
        {"continue": "home_intent_react", "done": "final_response"},
    )

    # Planner -> parallel_prefetch -> parallel_read_stage -> supervisor_observer
    # -> optional LLM route_plan supervision -> route dispatch.
    workflow.add_edge("home_intent_react", "planner")
    workflow.add_edge("planner", "parallel_prefetch")
    workflow.add_edge("parallel_prefetch", "parallel_read_stage")
    workflow.add_edge("parallel_read_stage", "supervisor_observer")
    workflow.add_edge("supervisor_observer", "llm_supervisor_route")

    # context_builder + skill_matcher run inside parallel_read_stage.
    workflow.add_conditional_edges(
        "llm_supervisor_route",
        dispatch_next_route_node,
        route_dests,
    )

    for agent_name in AGENT_NODE_NAMES:
        workflow.add_conditional_edges(
            agent_name,
            dispatch_next_route_node,
            route_dests,
        )

    workflow.add_conditional_edges(
        "evaluator",
        dispatch_after_evaluator,
        route_dests,
    )
    workflow.add_edge("final_response", END)

    if checkpointer is None:
        return workflow.compile()
    return workflow.compile(checkpointer=checkpointer)
