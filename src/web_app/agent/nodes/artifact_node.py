from src.web_app.agent.state import AgentState


def artifact_node(state: AgentState) -> AgentState:
    state["artifacts"] = state.get("artifacts", [])
    return state
