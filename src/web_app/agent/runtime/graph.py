"""Supervisor Agent Runtime — LangGraph multi-agent orchestration.

Routes user requests through a Planner, then executes the resulting
RoutePlan through conditional agent nodes (research, RAG, artifact,
MCP tool, memory, skill), concluding with an evaluator and final_response.
"""

from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.fallback import run_fallback
from src.web_app.agent.runtime.intent_schema import normalize_agent_name
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state import AgentRuntimeState, append_output


class AgentRuntime:
    def __init__(self, db: Session, payload: dict[str, Any]):
        self.db = db
        self.payload = payload
        self.nodes = RuntimeNodes(db, payload)

    async def run(self, state: AgentRuntimeState) -> AgentRuntimeState:
        graph = self._build_langgraph()
        if graph:
            return await graph.ainvoke(
                state,
                config={
                    "configurable": {
                        "thread_id": state.get("thread_id") or f"user:{state.get('user_id')}:run:{state.get('run_id')}",
                        "user_id": state.get("user_id"),
                        "run_id": state.get("run_id"),
                    }
                },
            )
        return await run_fallback(self._fallback_nodes(), state)

    def _fallback_nodes(self):
        return [
            self.nodes.permission_guard,
            self.nodes.home_intent_react,
            self.nodes.planner,
            self.nodes.context_builder,
            self.nodes.skill_matcher,
            self.nodes.research,
            self.nodes.rag,
            self.nodes.artifact,
            self.nodes.skill_librarian,
            self.nodes.tool,
            self.nodes.memory_writer,
            self.nodes.skill_draft_detector,
            self.nodes.evaluator,
            self.nodes.final_response,
        ]

    def _build_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        workflow = StateGraph(AgentRuntimeState)

        # ── Register all nodes ────────────────────────────────────
        workflow.add_node("permission_guard", self.nodes.permission_guard)
        workflow.add_node("home_intent_react", self.nodes.home_intent_react)
        workflow.add_node("planner", self.nodes.planner)
        workflow.add_node("context_builder", self.nodes.context_builder)
        workflow.add_node("skill_matcher", self.nodes.skill_matcher)
        workflow.add_node("research_agent", self.nodes.research_agent)
        workflow.add_node("rag_agent", self.nodes.rag_agent)
        workflow.add_node("artifact_agent", self.nodes.artifact_agent)
        workflow.add_node("tool_agent", self.nodes.tool_agent)
        workflow.add_node("memory_agent", self.nodes.memory_agent)
        workflow.add_node("skill_agent", self.nodes.skill_agent)
        workflow.add_node("evaluator", self.nodes.evaluator)
        workflow.add_node("final_response", self.nodes.final_response)

        # ── Edges ─────────────────────────────────────────────────
        workflow.set_entry_point("permission_guard")

        # After permission check: if blocked/approval → final_response, else → planner
        workflow.add_conditional_edges(
            "permission_guard",
            self._after_permission,
            {"continue": "home_intent_react", "done": "final_response"},
        )

        # Planner → context_builder
        workflow.add_edge("home_intent_react", "planner")
        workflow.add_edge("planner", "context_builder")

        # Context → skill_matcher
        workflow.add_edge("context_builder", "skill_matcher")

        # After skill_matcher: dispatch to first agent in route_plan
        workflow.add_conditional_edges(
            "skill_matcher",
            self._dispatch_next_route_node,
            {
                "research_agent": "research_agent",
                "rag_agent": "rag_agent",
                "artifact_agent": "artifact_agent",
                "tool_agent": "tool_agent",
                "memory_agent": "memory_agent",
                "skill_agent": "skill_agent",
                "evaluator": "evaluator",
                "final_response": "final_response",
            },
        )

        # After each agent: dispatch to next route node
        for agent_name in ("research_agent", "rag_agent", "artifact_agent",
                           "tool_agent", "memory_agent", "skill_agent"):
            workflow.add_conditional_edges(
                agent_name,
                self._dispatch_next_route_node,
                {
                    "research_agent": "research_agent",
                    "rag_agent": "rag_agent",
                    "artifact_agent": "artifact_agent",
                    "tool_agent": "tool_agent",
                    "memory_agent": "memory_agent",
                    "skill_agent": "skill_agent",
                    "evaluator": "evaluator",
                    "final_response": "final_response",
                },
            )

        # Evaluator → final_response (memory/skill agents run BEFORE evaluator)
        workflow.add_edge("evaluator", "final_response")

        # Final response → END
        workflow.add_edge("final_response", END)

        return workflow.compile()

    # ── Conditional routing ──────────────────────────────────────

    def _after_permission(self, state: AgentRuntimeState) -> str:
        if state.get("route") in {"approval", "blocked"} or state.get("error"):
            return "done"
        return "continue"

    def _dispatch_next_route_node(self, state: AgentRuntimeState) -> str:
        """Pop the next node from route_plan["route"] and return its name.
        Falls back to 'final_response' when no more nodes remain."""
        route_plan = state.get("route_plan") or {}
        route_list = list(route_plan.get("route", []))
        completed = list(state.get("completed_nodes", []))

        for node_name in route_list:
            if node_name not in completed:
                # Map route item names to registered agent node names
                return self._map_route_to_node(node_name)

        return "final_response"

    def _map_route_to_node(self, route_item: str) -> str:
        """Map a route_plan route item to a registered graph node name."""
        normalized = normalize_agent_name(route_item)
        mapping = {
            "context_builder": "context_builder",
            "skill_matcher": "skill_matcher",
            "research_agent": "research_agent",
            "rag_agent": "rag_agent",
            "artifact_agent": "artifact_agent",
            "tool_agent": "tool_agent",
            "memory_agent": "memory_agent",
            "skill_agent": "skill_agent",
            "evaluator": "evaluator",
            "final_response": "final_response",
        }
        return mapping.get(normalized, mapping.get(route_item, "final_response"))
