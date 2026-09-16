"""All observations return to the single native Supervisor."""
from langgraph.graph import START, END, StateGraph
from .state import AgentRuntimeState
from .chat_control import controlled_node


def build_graph(nodes, checkpointer=None):
    nodes.checkpoint_enabled = checkpointer is not None
    graph = StateGraph(AgentRuntimeState)
    names = ("permission_guard", "bootstrap_context", "supervisor", "capability", "deep_research", "tool_runtime", "document_read")
    for name in names:
        graph.add_node(name, controlled_node(name, getattr(nodes, name), nodes.db))
    graph.add_edge(START, "permission_guard")
    graph.add_edge("permission_guard", "bootstrap_context")
    graph.add_edge("bootstrap_context", "supervisor")
    def route(state):
        if state.get("status") in {"completed", "failed"}:
            return END
        action = state.get("current_action", {})
        if state.get("termination_reason"):
            return "supervisor"
        return {"deep_research": "deep_research", "tool": "tool_runtime", "document_read": "document_read"}.get(action.get("action"), "capability")
    graph.add_conditional_edges("supervisor", route, {END: END, **{n: n for n in names[2:]}})
    for name in names[3:]:
        graph.add_edge(name, "supervisor")
    return graph.compile(checkpointer=checkpointer)
