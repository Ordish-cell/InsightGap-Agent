from src.web_app.agent.state import AgentState


def memory_writer_node(state: AgentState) -> AgentState:
    state["memory_updates"] = state.get("memory_updates", [])
    return state
