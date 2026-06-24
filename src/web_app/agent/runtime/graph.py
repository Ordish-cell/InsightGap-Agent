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
    dispatch_after_evaluator,
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
    def __init__(self, db: Session, payload: dict[str, Any], stream_queue: Any = None):
        self.db = db
        self.payload = payload
        self._stream_queue = stream_queue
        self.nodes = RuntimeNodes(db, payload, stream_queue)

    async def run(self, state: AgentRuntimeState) -> AgentRuntimeState:
        import logging
        _run_log = logging.getLogger(__name__)
        # Remove non-serializable objects before LangGraph sees the state.
        state.pop("_stream_queue", None)
        graph = await self._build_langgraph()
        cfg = build_langgraph_invoke_config(state)
        _run_log.info(
            "[CHECKPOINTER] graph.ainvoke thread_id=%s run_id=%s",
            cfg.get("configurable", {}).get("thread_id"), state.get("run_id"),
        )
        if graph:
            return await graph.ainvoke(state, config=cfg)
        return await run_fallback(self._fallback_nodes(), state)

    async def resume_from_interrupt(
        self,
        resume_payload: dict[str, Any],
        thread_id: str,
    ) -> AgentRuntimeState:
        """Resume a graph that paused via LangGraph interrupt().

        Uses Command(resume=resume_payload) to continue from the
        checkpoint saved at the interrupt() call site.  Does NOT
        re-run the graph from entry_point.

        Args:
            resume_payload: The value that interrupt() will return
                inside the paused node (must contain "action" key).
            thread_id: The same thread_id used when the graph was
                first invoked (checkpoint key).

        Raises:
            RuntimeError: If checkpointer is not enabled or
                thread_id is missing.
        """
        import logging
        _log = logging.getLogger(__name__)
        from langgraph.types import Command

        if not thread_id:
            raise RuntimeError(
                "resume_from_interrupt requires a thread_id "
                "(checkpoint key). The original run must set "
                "state['thread_id'] = f'run:{run_id}'."
            )
        if not getattr(settings, "agent_langgraph_checkpointer_enabled", False):
            raise RuntimeError(
                "resume_from_interrupt requires checkpointer to be "
                "enabled (agent_langgraph_checkpointer_enabled=true). "
                "Without a checkpointer, LangGraph cannot save/restore "
                "state at the interrupt point."
            )

        action = resume_payload.get("action", "unknown")
        tool_call_id = resume_payload.get("tool_call_id")
        _log.info(
            "[approval_interrupt_resume] thread_id=%s action=%s "
            "tool_call_id=%s",
            thread_id, action, tool_call_id,
        )

        graph = await self._build_langgraph()
        if not graph:
            raise RuntimeError("LangGraph is not available")

        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(Command(resume=resume_payload), config=config)
        _log.info(
            "[approval_interrupt_resume] completed "
            "thread_id=%s action=%s status=%s",
            thread_id, action, result.get("status") if isinstance(result, dict) else "N/A",
        )
        return result

    def _fallback_nodes(self):
        return build_fallback_nodes(self.nodes)

    async def _build_langgraph(self):
        import logging
        _build_log = logging.getLogger(__name__)
        checkpointer = None
        if getattr(settings, "agent_langgraph_checkpointer_enabled", False):
            backend = getattr(settings, "agent_checkpointer_backend", "postgres")
            require_durable = getattr(settings, "agent_checkpointer_require_durable", False)
            cp_conn_string = getattr(settings, "agent_checkpointer_database_url", "") or getattr(settings, "database_url", "").replace("+psycopg2", "")

            # ── postgres: use AsyncPostgresSaver for graph.ainvoke() ──
            if backend == "postgres" and cp_conn_string:
                from src.web_app.agent.runtime.checkpointers import _AsyncPostgresSaverHandle
                try:
                    checkpointer = await _AsyncPostgresSaverHandle.create(cp_conn_string)
                    _build_log.info(
                        "[CHECKPOINTER] backend=postgres saver_type=%s durable=True",
                        type(checkpointer).__name__,
                    )
                except Exception as exc:
                    if require_durable:
                        raise RuntimeError(
                            f"[CHECKPOINTER] backend=postgres unavailable: {exc}"
                        ) from exc
                    _build_log.warning(
                        "[CHECKPOINTER] AsyncPostgresSaver unavailable — "
                        "falling back to memory. error=%s", exc
                    )
                    checkpointer = build_checkpointer(backend="memory", require_durable=False)
            else:
                checkpointer = build_checkpointer(
                    backend=backend,
                    conn_string=cp_conn_string,
                    redis_url=getattr(settings, "redis_url", None),
                    redis_password=getattr(settings, "redis_password", ""),
                    redis_key_prefix=getattr(settings, "redis_checkpointer_key_prefix", "langgraph:checkpoint:"),
                    require_durable=require_durable,
                )
                cp_type = type(checkpointer).__name__ if checkpointer else "None"
                _build_log.info(
                    "[CHECKPOINTER] backend=%s checkpointer=%s durable=%s",
                    backend, cp_type, require_durable or backend != "memory",
                )
        else:
            _build_log.info(
                "[CHECKPOINTER] checkpointer disabled (agent_langgraph_checkpointer_enabled=False)"
            )
        return build_agent_runtime_graph(
            self.nodes,
            after_permission=self._after_permission,
            dispatch_next_route_node=self._dispatch_next_route_node,
            dispatch_after_evaluator=self._dispatch_after_evaluator,
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

    def _dispatch_after_evaluator(self, state: AgentRuntimeState) -> str:
        return dispatch_after_evaluator(state)

    def _record_supervisor_dispatch_audit(self, state: AgentRuntimeState, legacy_next_node: str) -> str:
        return record_supervisor_dispatch_audit(state, legacy_next_node)

    def _map_route_to_node(self, route_item: str) -> str:
        """Map a route_plan route item to a registered graph node name."""
        return map_route_to_node(route_item)
