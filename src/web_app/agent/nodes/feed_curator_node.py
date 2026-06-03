from src.web_app.agent.state import AgentState


def feed_curator_node(state: AgentState) -> AgentState:
    state["feed_cards"] = []
    return state
