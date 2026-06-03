from src.web_app.agent.adapters.open_deep_research_adapter import OpenDeepResearchAdapter
from src.web_app.agent.state import AgentState


def research_node(state: AgentState) -> AgentState:
    state["research"] = OpenDeepResearchAdapter().run_research(state.get("user_input", ""), state.get("context", {}), state.get("user_id", 1), {})
    return state
