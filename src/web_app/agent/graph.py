from typing import Any

from src.web_app.agent.nodes.context_builder_node import context_builder_node
from src.web_app.agent.nodes.evaluator_node import evaluator_node
from src.web_app.agent.nodes.permission_guard_node import permission_guard_node
from src.web_app.agent.nodes.router_node import router_node
from src.web_app.agent.state import AgentState


def run_agent_graph(payload: dict[str, Any]) -> AgentState:
    state: AgentState = {
        "user_id": payload.get("user_id", 1),
        "run_id": payload.get("run_id", 0),
        "user_input": payload.get("user_input", ""),
        "mode": payload.get("mode", "react"),
        "tool_calls": payload.get("tool_calls", []),
    }
    for node in (router_node, permission_guard_node, context_builder_node, evaluator_node):
        state = node(state)
    state["final_output"] = state.get("final_output") or "Agent runtime skeleton completed."
    return state
