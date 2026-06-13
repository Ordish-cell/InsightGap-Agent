"""Supervisor Agent Runtime - LangGraph multi-agent orchestration.

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

from src.web_app.agent.runtime.checkpointers import build_checkpointer
from src.web_app.agent.runtime.dispatch import (
    END_SENTINEL as _END_SENTINEL,
    after_permission,
    dispatch_next_route_node,
    map_route_to_node,
    record_supervisor_dispatch_audit,
)
from src.web_app.agent.runtime.fallback import run_fallback
from src.web_app.agent.runtime.graph_builder import build_agent_runtime_graph
from src.web_app.agent.runtime.graph_config import build_langgraph_invoke_config
from src.web_app.agent.runtime.graph_registry import build_fallback_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state import AgentRuntimeState
from src.web_app.core.config import settings


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
                config=build_langgraph_invoke_config(state),
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
        # Purge stale pause artifacts.
        state["approval_payload"] = None
        state["route"] = ""  # prevent evaluator from reverting to waiting_approval
        state["error"] = state.get("_tool_error") or ""  # only carry real tool errors
        # Clean tool_call error so "approval_required" doesn't leak as failure reason
        tc = state.get("tool_call") or {}
        if isinstance(tc, dict):
            tc["error"] = ""  # always clean; never carry approval_required forward
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
        return build_fallback_nodes(self.nodes)

    def _build_langgraph(self):
        checkpointer = None
        if getattr(settings, "agent_langgraph_checkpointer_enabled", False):
            checkpointer = build_checkpointer(getattr(settings, "redis_url", None))
        return build_agent_runtime_graph(
            self.nodes,
            after_permission=self._after_permission,
            dispatch_next_route_node=self._dispatch_next_route_node,
            checkpointer=checkpointer,
        )

    # Conditional routing.

    def _after_permission(self, state: AgentRuntimeState) -> str:
        return after_permission(state)

    def _dispatch_next_route_node(self, state: AgentRuntimeState) -> str:
        """Pop the next node from route_plan["route"] and return its name.

        When the run is waiting for approval, the graph performs a true
        interrupt: we return __end__ which maps to END, terminating the
        graph immediately.  No evaluator, no final_response - the
        agent_service layer detects the paused state and emits
        approval_required / run_paused.

        Falls back to 'final_response' when all route nodes are completed.
        """
        return dispatch_next_route_node(state)

    def _record_supervisor_dispatch_audit(self, state: AgentRuntimeState, legacy_next_node: str) -> str:
        return record_supervisor_dispatch_audit(state, legacy_next_node)

    def _map_route_to_node(self, route_item: str) -> str:
        """Map a route_plan route item to a registered graph node name."""
        return map_route_to_node(route_item)
