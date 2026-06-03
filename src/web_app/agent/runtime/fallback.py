from src.web_app.agent.runtime.state import AgentRuntimeState


async def run_fallback(nodes, state: AgentRuntimeState) -> AgentRuntimeState:
    for node in nodes:
        state = await node(state)
        if state.get("route") in {"approval", "blocked"} or state.get("error"):
            break
    return state
