from src.web_app.agent.state import AgentState
from src.web_app.context.builder import ContextBuilder


def context_builder_node(state: AgentState) -> AgentState:
    state["context"] = ContextBuilder().build({"task": state.get("user_input", ""), "tool_state": state.get("tool_calls", [])})
    return state
