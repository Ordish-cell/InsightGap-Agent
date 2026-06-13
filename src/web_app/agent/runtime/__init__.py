__all__ = ["AgentRuntime"]


def __getattr__(name):
    if name == "AgentRuntime":
        from src.web_app.agent.runtime.graph import AgentRuntime
        return AgentRuntime
    raise AttributeError(name)
