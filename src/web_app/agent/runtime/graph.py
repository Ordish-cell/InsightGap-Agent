"""Supervisor Agent Runtime — LangGraph multi-agent orchestration.

Routes user requests through a Planner, then executes the resulting
RoutePlan through conditional agent nodes (research, RAG, artifact,
MCP tool, memory, skill), concluding with an evaluator and final_response.

When a tool requires approval (L3/L4), the graph performs a true interrupt:
tool_agent sets status=waiting_approval, the dispatcher routes to END,
and the graph terminates cleanly.  The agent_service layer detects the
pause, emits approval events, and later resumes the run by re-invoking
the graph with the pre-executed tool result injected.
"""

from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.fallback import run_fallback
from src.web_app.agent.runtime.intent_schema import normalize_agent_name
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state import AgentRuntimeState, append_output

# Sentinel key used by conditional edges to terminate the graph immediately
# (true interrupt — no evaluator, no final_response).
_END_SENTINEL = "__end__"


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

    async def resume_from_approval(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Re-run the graph after an approval has been granted/rejected."""
        import logging
        _log = logging.getLogger(__name__)

        tc_before = state.get("tool_call") or {}
        _log.info(
            "[approval_resume_debug] runtime=start "
            "status=%s approval_required=%s route=%s error=%s "
            "tool_call.error=%s pending_approval_id=%s pending_tcid=%s "
            "resolved_tool_call_ids=%s _resume_context=%s completed_nodes=%s",
            state.get("status"), state.get("approval_required"),
            state.get("route"), state.get("error"),
            tc_before.get("error") if isinstance(tc_before, dict) else "N/A",
            state.get("pending_approval_id"), state.get("pending_tool_call_id"),
            state.get("resolved_tool_call_ids"), state.get("_resume_context"),
            state.get("completed_nodes"),
        )

        state.setdefault("completed_nodes", [])
        state.setdefault("resolved_tool_call_ids", [])
        state["status"] = "resuming"
        state["approval_required"] = False
        # ── Purge stale pause artifacts ──────────────────────────
        state["approval_payload"] = None
        state["route"] = ""  # prevent evaluator from reverting to waiting_approval
        state["error"] = state.get("_tool_error") or ""  # only carry real tool errors
        # Clean tool_call error so "approval_required" doesn't leak as failure reason
        tc = state.get("tool_call") or {}
        if isinstance(tc, dict):
            tc["error"] = ""  # always clean — never carry approval_required forward
            state["tool_call"] = tc

        _log.info(
            "[approval_resume_debug] runtime=after_cleanup "
            "status=%s error=%s route=%s approval_required=%s",
            state.get("status"), state.get("error"), state.get("route"),
            state.get("approval_required"),
        )

        result = await self.run(state)

        tc_after = result.get("tool_call") or {}
        _log.info(
            "[approval_resume_debug] runtime=after_graph "
            "result.status=%s error=%s route=%s approval_required=%s "
            "tool_call.error=%s _resume_context=%s resolved_ids=%s "
            "final_answer_preview=%s",
            result.get("status"), result.get("error"), result.get("route"),
            result.get("approval_required"),
            tc_after.get("error") if isinstance(tc_after, dict) else "N/A",
            result.get("_resume_context"), result.get("resolved_tool_call_ids"),
            (result.get("final_answer") or "")[:120],
        )
        return result

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

        # ── Shared route destinations (including END for true interrupt) ──
        _route_dests = {
            "research_agent": "research_agent",
            "rag_agent": "rag_agent",
            "artifact_agent": "artifact_agent",
            "tool_agent": "tool_agent",
            "memory_agent": "memory_agent",
            "skill_agent": "skill_agent",
            "evaluator": "evaluator",
            "final_response": "final_response",
            _END_SENTINEL: END,
        }

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
            _route_dests,
        )

        # After each agent: dispatch to next route node (or END on interrupt)
        for agent_name in ("research_agent", "rag_agent", "artifact_agent",
                           "tool_agent", "memory_agent", "skill_agent"):
            workflow.add_conditional_edges(
                agent_name,
                self._dispatch_next_route_node,
                _route_dests,
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

        When the run is waiting for approval, the graph performs a true
        interrupt: we return __end__ which maps to END, terminating the
        graph immediately.  No evaluator, no final_response — the
        agent_service layer detects the paused state and emits
        approval_required / run_paused.

        Falls back to 'final_response' when all route nodes are completed.
        """
        if state.get("status") == "waiting_approval":
            return _END_SENTINEL  # true interrupt → END

        route_plan = state.get("route_plan") or {}
        route_list = list(route_plan.get("route", []))
        completed = list(state.get("completed_nodes", []))

        for node_name in route_list:
            if node_name not in completed:
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
