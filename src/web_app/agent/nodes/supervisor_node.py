from src.web_app.agent.state import AgentState


def supervisor_node(state: AgentState) -> AgentState:
    state["agent_name"] = "research_agent" if state.get("intent") == "deep_research" else "assistant_agent"
    return state
