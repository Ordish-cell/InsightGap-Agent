"""Memory graph rebuild contracts and separately configured Redis probes.

Sharing InMemorySaver verifies graph reconstruction only. Real process and
owned PostgreSQL service restart are covered by test_supervisor_process_restart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from typing import TypedDict


class _TestState(TypedDict):
    x: int
    status: str
    run_id: int
    thread_id: str


def _node(state):
    result = interrupt({"type": "approval_required", "approval_id": 999, "msg": "pause"})
    state["status"] = result.get("action", "unknown")
    state["x"] = result.get("value", 0)
    return state


# ── shared checkpointer ─────────────────────────────────────────────

_SHARED_SAVER = InMemorySaver()


def _build_graph():
    g = StateGraph(_TestState)
    g.add_node("n", _node)
    g.set_entry_point("n")
    g.add_edge("n", END)
    return g.compile(checkpointer=_SHARED_SAVER)


# ── tests ────────────────────────────────────────────────────────────


class TestCrossInstanceRestart:
    """Rebuild graph objects in one process using the same in-memory saver."""

    def test_pause_and_resume_same_saver(self):
        """Same InMemorySaver: pause in graph A, resume in graph B."""
        app_a = _build_graph()
        tid = "e2e-restart-approved-800"
        cfg = {"configurable": {"thread_id": tid}}
        state = {"x": 0, "status": "running", "run_id": 800, "thread_id": tid}

        # Phase 1: Pause
        r1 = app_a.invoke(state, config=cfg)
        interrupt_data = r1.get("__interrupt__")
        assert interrupt_data is not None, f"Expected __interrupt__, got keys={list(r1.keys())}"
        assert interrupt_data[0].value["type"] == "approval_required"

        # Phase 2: New graph (simulated restart), same checkpointer
        app_b = _build_graph()
        r2 = app_b.invoke(
            Command(resume={"action": "approved", "value": 42}),
            config=cfg,
        )
        assert r2.get("status") == "approved"
        assert r2.get("x") == 42

    def test_reject_cross_instance_resume(self):
        """Reject path: pause in graph A, reject resume in graph B."""
        app_a = _build_graph()
        tid = "e2e-restart-rejected-801"
        cfg = {"configurable": {"thread_id": tid}}
        state = {"x": 0, "status": "running", "run_id": 801, "thread_id": tid}

        # Pause
        r1 = app_a.invoke(state, config=cfg)
        assert "__interrupt__" in r1

        # Reject in new graph
        app_b = _build_graph()
        r2 = app_b.invoke(
            Command(resume={"action": "rejected", "reason": "user declined"}),
            config=cfg,
        )
        assert r2.get("status") == "rejected"
        assert r2.get("x") == 0

    def test_thread_id_stability(self):
        """thread_id remains stable across restart instances."""
        app_a = _build_graph()
        tid = "e2e-restart-tid-802"
        cfg = {"configurable": {"thread_id": tid}}

        app_a.invoke(
            {"x": 0, "status": "running", "run_id": 802, "thread_id": tid},
            config=cfg,
        )
        app_b = _build_graph()
        r2 = app_b.invoke(
            Command(resume={"action": "approved", "value": 99}),
            config=cfg,
        )
        assert r2.get("thread_id") == tid


class TestRedisBackendReadiness:
    """Probe configured Redis availability; this does not establish V2 support."""

    def test_redis_saver_can_connect(self):
        """RedisSaver constructor succeeds with real Redis (backend=redis)."""
        from src.web_app.agent.runtime.checkpointers import build_checkpointer
        REDIS_URL = os.environ.get(
            "REDIS_URL", "redis://:your_password@redis_host:6379/0"
        )
        REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "your_password")

        try:
            saver = build_checkpointer(
                backend="redis",
                redis_url=REDIS_URL,
                redis_password=REDIS_PASSWORD,
                require_durable=True,
            )
        except RuntimeError:
            pytest.skip("Redis not reachable (network)")
        assert saver is not None
        assert "Redis" in type(saver).__name__

    def test_redis_fail_fast_when_unreachable(self):
        """require_durable=True + unreachable Redis raises RuntimeError."""
        from src.web_app.agent.runtime.checkpointers import build_checkpointer
        with pytest.raises(RuntimeError, match="backend=redis unavailable"):
            build_checkpointer(
                backend="redis",
                redis_url="redis://192.168.99.99:99999/0",
                require_durable=True,
            )

    def test_sync_invoke_interrupt_pattern(self):
        """Verify the LangGraph 1.x sync invoke/interrupt/resume API works.
        This is the pattern used by InMemorySaver and all other backends."""
        saver = InMemorySaver()
        g = StateGraph(_TestState)
        g.add_node("n", _node)
        g.set_entry_point("n")
        g.add_edge("n", END)
        app = g.compile(checkpointer=saver)

        cfg = {"configurable": {"thread_id": "sync-test-1"}}
        # Pause
        r1 = app.invoke(
            {"x": 0, "status": "running", "run_id": 1, "thread_id": "sync-test-1"},
            config=cfg,
        )
        assert "__interrupt__" in r1

        # Resume
        r2 = app.invoke(Command(resume={"action": "approved", "value": 100}), config=cfg)
        assert r2["x"] == 100
        assert r2["status"] == "approved"
