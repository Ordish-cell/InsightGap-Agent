from src.web_app.agent.state import AgentState


def router_node(state: AgentState) -> AgentState:
    text = state.get("user_input", "")
    state["intent"] = "deep_research" if "research" in text.lower() or "研究" in text else "chat"
    return state
