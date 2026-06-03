from src.web_app.agent.state import AgentState


def evaluator_node(state: AgentState) -> AgentState:
    state["final_output"] = "Waiting for approval." if state.get("approvals") else "Ready."
    return state
