from src.web_app.agent.state import AgentState


def skill_librarian_node(state: AgentState) -> AgentState:
    state["skill_drafts"] = state.get("skill_drafts", [])
    return state
