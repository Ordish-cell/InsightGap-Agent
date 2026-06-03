from src.web_app.agent.state import AgentState
from src.web_app.services.rag_service import rag_service


def rag_node(state: AgentState) -> AgentState:
    state["evidence"] = rag_service.search(state.get("user_id", 1), state.get("user_input", "")).get("evidence", [])
    return state
