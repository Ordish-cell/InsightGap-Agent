from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.fallback import run_fallback
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state import AgentRuntimeState


class AgentRuntime:
    def __init__(self, db: Session, payload: dict[str, Any]):
        self.db = db
        self.payload = payload
        self.nodes = RuntimeNodes(db, payload)

    async def run(self, state: AgentRuntimeState) -> AgentRuntimeState:
        graph = self._build_langgraph()
        if graph:
            return await graph.ainvoke(state)
        return await run_fallback(self._fallback_nodes(), state)

    def _fallback_nodes(self):
        return [
            self.nodes.permission_guard,
            self.nodes.router,
            self.nodes.context_builder,
            self.nodes.research,
            self.nodes.rag,
            self.nodes.artifact,
            self.nodes.skill_librarian,
            self.nodes.tool,
            self.nodes.memory_writer,
            self.nodes.evaluator,
        ]

    def _build_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        workflow = StateGraph(AgentRuntimeState)
        workflow.add_node("permission_guard", self.nodes.permission_guard)
        workflow.add_node("router", self.nodes.router)
        workflow.add_node("context_builder", self.nodes.context_builder)
        workflow.add_node("research", self.nodes.research)
        workflow.add_node("rag", self.nodes.rag)
        workflow.add_node("artifact", self.nodes.artifact)
        workflow.add_node("skill_librarian", self.nodes.skill_librarian)
        workflow.add_node("tool", self.nodes.tool)
        workflow.add_node("memory_writer", self.nodes.memory_writer)
        workflow.add_node("evaluator", self.nodes.evaluator)

        workflow.set_entry_point("permission_guard")
        workflow.add_conditional_edges("permission_guard", self._after_permission, {"continue": "router", "done": "evaluator"})
        workflow.add_edge("router", "context_builder")
        workflow.add_conditional_edges("context_builder", self._route_after_context, {"research": "research", "rag": "rag", "artifact": "artifact", "skill": "skill_librarian", "tool": "tool", "memory": "memory_writer"})
        workflow.add_edge("research", "memory_writer")
        workflow.add_edge("rag", "memory_writer")
        workflow.add_edge("artifact", "memory_writer")
        workflow.add_edge("skill_librarian", "memory_writer")
        workflow.add_edge("tool", "memory_writer")
        workflow.add_edge("memory_writer", "evaluator")
        workflow.add_edge("evaluator", END)
        return workflow.compile()

    def _after_permission(self, state: AgentRuntimeState) -> str:
        return "done" if state.get("route") in {"approval", "blocked"} or state.get("error") else "continue"

    def _route_after_context(self, state: AgentRuntimeState) -> str:
        route = state.get("route", "memory")
        return route if route in {"research", "rag", "artifact", "skill", "tool", "memory"} else "memory"
