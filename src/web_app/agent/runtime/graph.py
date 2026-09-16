"""Single Supervisor runtime with versioned durable checkpoint validation."""

from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpointers import build_checkpointer
from src.web_app.agent.runtime.graph_config import build_langgraph_invoke_config
from src.web_app.agent.runtime.state import AgentRuntimeState
from src.web_app.core.config import settings


class AgentRuntime:
    def __init__(self, db: Session, payload: dict[str, Any], stream_queue: Any = None):
        self.db = db
        self.payload = payload
        self._stream_queue = stream_queue
        self._checkpointer = None
        if payload.get("runtime_version", 2) != 2:
            raise ValueError("RUNTIME_VERSION_UNSUPPORTED: only Supervisor runtime 2 can execute")
        from src.web_app.agent.runtime.nodes import SupervisorNodes
        self.nodes = SupervisorNodes(db, payload, stream_queue)

    async def run(self, state: AgentRuntimeState) -> AgentRuntimeState:
        import logging
        _run_log = logging.getLogger(__name__)
        # Remove non-serializable objects before LangGraph sees the state.
        state.pop("_stream_queue", None)
        if state.get("runtime_version", 2) != 2:
            raise ValueError("RUNTIME_VERSION_UNSUPPORTED: only Supervisor runtime 2 can execute")
        state["runtime_version"] = 2
        state["loop_protocol_version"] = 1
        state.setdefault("interaction_version", 2)
        try:
            graph = await self._build_langgraph()
            cfg = build_langgraph_invoke_config(state)
            _run_log.info(
                "[CHECKPOINTER] graph.ainvoke thread_id=%s run_id=%s",
                cfg.get("configurable", {}).get("thread_id"), state.get("run_id"),
            )
            if graph:
                cfg["recursion_limit"] = max(50, settings.agent_max_supervisor_steps * 3 + 10)
                result = await graph.ainvoke(state, config=cfg)
                return self._project_interrupt(result)
            raise RuntimeError("LangGraph is required for the Supervisor runtime")
        finally:
            await self._close_checkpointer()

    async def resume_from_interrupt(
        self,
        resume_payload: dict[str, Any],
        thread_id: str,
    ) -> AgentRuntimeState:
        """Resume a graph that paused via LangGraph interrupt().

        Uses Command(resume=resume_payload) to continue from the
        saved checkpoint. The interrupted node starts again from its beginning;
        operations preceding interrupt must therefore be replay-safe. Earlier
        completed graph nodes do not restart from the graph entry point.

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

        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        config["recursion_limit"] = max(50, settings.agent_max_supervisor_steps * 3 + 10)
        try:
            graph = await self._build_langgraph()
            if not graph:
                raise RuntimeError("LangGraph is not available")
            snapshot = await graph.aget_state(config)
            if not snapshot or (snapshot.values or {}).get("runtime_version") != 2:
                raise ValueError("RUNTIME_VERSION_UNSUPPORTED: checkpoint is missing or belongs to the retired runtime")
            if snapshot.values.get("loop_protocol_version") != 1:
                raise ValueError("LOOP_PROTOCOL_UNSUPPORTED: checkpoint predates the native Supervisor loop")
            result = self._project_interrupt(await graph.ainvoke(Command(resume=resume_payload), config=config))
        finally:
            await self._close_checkpointer()
        _log.info(
            "[approval_interrupt_resume] completed "
            "thread_id=%s action=%s status=%s",
            thread_id, action, result.get("status") if isinstance(result, dict) else "N/A",
        )
        return result

    async def _close_checkpointer(self):
        saver, self._checkpointer = self._checkpointer, None
        ctx = getattr(saver, "_checkpointer_ctx", None)
        if ctx is not None:
            try:
                if hasattr(ctx, "__aexit__"):
                    await ctx.__aexit__(None, None, None)
                else:
                    ctx.__exit__(None, None, None)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Checkpoint connection cleanup failed")

    def _project_interrupt(self, state):
        interrupts = state.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            state.update(status="waiting_approval", approval_required=True, approval_pause_mode="interrupt",
                tool_call={"id": payload["tool_call_id"], "tool_name": payload["tool_name"], "status": "waiting_approval"},
                pending_approval_id=str(payload["approval_id"]), pending_tool_call_id=payload["tool_call_id"],
                pending_tool_name=payload["tool_name"], approval_payload=payload,
                pending_tool_args=(state.get("current_action", {}).get("arguments", {}).get("input", {})))
        state.pop("__interrupt__", None)
        return state

    async def _build_langgraph(self):
        import logging
        _build_log = logging.getLogger(__name__)
        checkpointer = None
        if getattr(settings, "agent_langgraph_checkpointer_enabled", False):
            backend = getattr(settings, "agent_checkpointer_backend", "postgres")
            if backend == "redis":
                raise RuntimeError(
                    "Supervisor V2 does not support the current Redis checkpoint adapter; "
                    "use AGENT_CHECKPOINTER_BACKEND=postgres for durable approval recovery."
                )
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
        self._checkpointer = checkpointer
        from src.web_app.agent.runtime.graph_builder import build_graph
        return build_graph(self.nodes, checkpointer)
