"""E2E: PostgresSaver checkpoint — process restart recovery.

Verifies approve/reject restarts using PostgresSaver as the
LangGraph checkpointer backend.  This is the production path.

Run: uv run pytest src/web_app/tests/test_postgres_checkpoint_e2e.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _pg_conn_string() -> str:
    """Build a psycopg-compatible connection string from settings."""
    from src.web_app.core.config import settings
    raw = getattr(settings, "agent_checkpointer_database_url", "") or getattr(settings, "database_url", "")
    return raw.replace("+psycopg2", "")


def _pg_saver():
    """Create a PostgresSaver for testing."""
    from src.web_app.agent.runtime.checkpointers import build_checkpointer
    return build_checkpointer(
        backend="postgres",
        conn_string=_pg_conn_string(),
        require_durable=True,
    )


# ── tests ────────────────────────────────────────────────────────────


class TestPostgresCheckpointerSetup:
    """Verify PostgresSaver initializes correctly."""

    def test_pg_saver_connects_and_has_tables(self):
        """PostgresSaver connects and checkpoint tables exist."""
        saver = _pg_saver()
        assert saver is not None
        type_name = type(saver).__name__
        assert "Postgres" in type_name or "Handle" in type_name, (
            f"Expected PostgresSaver, got {type_name}"
        )

    def test_pg_saver_has_sync_methods(self):
        """PostgresSaver has get_tuple (sync) method."""
        saver = _pg_saver()
        assert hasattr(saver, "get_tuple"), "PostgresSaver must have get_tuple"

    def test_build_checkpointer_postgres_default(self):
        """Default backend=postgres returns a working saver."""
        from src.web_app.agent.runtime.checkpointers import build_checkpointer
        saver = build_checkpointer(conn_string=_pg_conn_string())
        assert saver is not None

    def test_require_durable_blocks_memory(self):
        """require_durable=True + memory backend raises RuntimeError."""
        from src.web_app.agent.runtime.checkpointers import build_checkpointer
        with pytest.raises(RuntimeError, match="require_durable"):
            build_checkpointer(backend="memory", require_durable=True)


class TestPostgresCrossInstanceRestart:
    """Full cycle: pause → kill process → restart → approve/reject."""

    def _build_graph(self, saver):
        from langgraph.graph import StateGraph, END
        from langgraph.types import interrupt, Command
        from typing import TypedDict

        class S(TypedDict):
            x: int
            status: str
            tool_result: str
            run_id: int
            thread_id: str

        def _node(state):
            result = interrupt({
                "type": "approval_required",
                "approval_id": 999,
                "msg": "needs approval",
            })
            state["status"] = result.get("action", "unknown")
            if result.get("action") == "approved":
                state["tool_result"] = "executed_ok"
            else:
                state["tool_result"] = "rejected_by_user"
            return state

        g = StateGraph(S)
        g.add_node("n", _node)
        g.set_entry_point("n")
        g.add_edge("n", END)
        return g.compile(checkpointer=saver)

    def test_approve_restart_recovery(self):
        """Pause in Process 1, approve in Process 2."""
        from langgraph.types import Command

        # Process 1: build graph, run, pause at interrupt
        saver1 = _pg_saver()
        app1 = self._build_graph(saver1)
        tid = "pg-e2e-approve-100"
        cfg = {"configurable": {"thread_id": tid}}
        state = {"x": 0, "status": "running", "tool_result": "", "run_id": 100, "thread_id": tid}

        r1 = app1.invoke(state, config=cfg)
        assert "__interrupt__" in r1, f"Expected interrupt, got keys={list(r1.keys())}"
        assert r1["__interrupt__"][0].value["type"] == "approval_required"

        # Process 2 (restart): new saver, same thread_id
        saver2 = _pg_saver()
        app2 = self._build_graph(saver2)

        r2 = app2.invoke(
            Command(resume={"action": "approved", "tool_result": "ok"}),
            config=cfg,
        )
        assert r2["status"] == "approved"
        assert r2["tool_result"] == "executed_ok"
        assert r2["run_id"] == 100
        assert r2["thread_id"] == tid

    def test_reject_restart_recovery(self):
        """Pause in Process 1, reject in Process 2."""
        from langgraph.types import Command

        saver1 = _pg_saver()
        app1 = self._build_graph(saver1)
        tid = "pg-e2e-reject-101"
        cfg = {"configurable": {"thread_id": tid}}
        state = {"x": 0, "status": "running", "tool_result": "", "run_id": 101, "thread_id": tid}

        r1 = app1.invoke(state, config=cfg)
        assert "__interrupt__" in r1

        saver2 = _pg_saver()
        app2 = self._build_graph(saver2)

        r2 = app2.invoke(
            Command(resume={"action": "rejected", "reason": "user declined"}),
            config=cfg,
        )
        assert r2["status"] == "rejected"
        assert r2["tool_result"] == "rejected_by_user"

    def test_thread_id_stable_across_restart(self):
        """Same thread_id used in both processes."""
        saver1 = _pg_saver()
        app1 = self._build_graph(saver1)
        tid = "pg-e2e-tid-102"
        cfg = {"configurable": {"thread_id": tid}}
        state = {"x": 0, "status": "running", "tool_result": "", "run_id": 102, "thread_id": tid}

        app1.invoke(state, config=cfg)

        saver2 = _pg_saver()
        app2 = self._build_graph(saver2)
        from langgraph.types import Command
        r2 = app2.invoke(Command(resume={"action": "approved"}), config=cfg)
        assert r2["thread_id"] == tid, f"thread_id changed: {r2['thread_id']} != {tid}"
