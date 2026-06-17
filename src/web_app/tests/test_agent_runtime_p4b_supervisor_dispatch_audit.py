import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph import _END_SENTINEL, AgentRuntime
from src.web_app.agent.runtime.supervisor import audit_supervisor_dispatch
from src.web_app.tests.db_test_utils import make_test_session


def _runtime() -> AgentRuntime:
    return AgentRuntime(make_test_session(), {})


def test_dispatch_audit_ok_when_supervisor_matches_legacy_next_node():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent", "final_response"]},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "next_expected_node": "rag_agent",
            "observed_pending_nodes": ["rag_agent", "final_response"],
            "observed_completed_nodes": [],
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["supervisor_dispatch_audit"]["status"] == "ok"
    assert state["supervisor_dispatch_audit"]["matched"] is True
    assert state["supervisor_dispatch_warnings"] == []


def test_dispatch_audit_mismatch_does_not_change_legacy_return():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent", "final_response"]},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "next_expected_node": "evaluator",
            "observed_pending_nodes": ["evaluator"],
            "observed_completed_nodes": [],
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["supervisor_dispatch_audit"]["status"] == "mismatch"
    assert state["supervisor_dispatch_audit"]["expected_next_node"] == "evaluator"
    assert state["supervisor_dispatch_audit"]["legacy_next_node"] == "rag_agent"
    assert "supervisor_dispatch_mismatch" in state["supervisor_dispatch_warnings"]


def test_dispatch_audit_skips_when_supervisor_decision_missing():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"]},
        "completed_nodes": [],
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["supervisor_dispatch_audit"]["status"] == "skipped"
    assert state["supervisor_dispatch_audit"]["reason"] == "missing_supervisor_decision"
    assert state["supervisor_dispatch_warnings"] == []


def test_waiting_approval_still_returns_end_sentinel_and_preserves_pause_fields():
    approval_payload = {"approval_id": "a1"}
    state = {
        "status": "waiting_approval",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"]},
        "completed_nodes": [],
        "approval_payload": approval_payload,
        "pending_tool_call_id": 7,
        "final_payload": {"answer": "paused"},
        "final_output": "paused",
        "supervisor_decision": {
            "mode": "observe_only",
            "next_expected_node": "tool_agent",
            "waiting_approval": True,
        },
    }
    protected = {
        key: copy.deepcopy(state.get(key))
        for key in (
            "route_plan",
            "completed_nodes",
            "status",
            "approval_payload",
            "pending_tool_call_id",
            "final_payload",
            "final_output",
        )
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == _END_SENTINEL
    assert state["supervisor_dispatch_audit"]["status"] == "skipped"
    assert state["supervisor_dispatch_audit"]["reason"] == "waiting_approval"
    for key, value in protected.items():
        assert state.get(key) == value


def test_route_completed_returns_final_response_and_audit_does_not_block():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"]},
        "completed_nodes": ["rag_agent"],
        "supervisor_decision": {
            "mode": "observe_only",
            "next_expected_node": None,
            "observed_pending_nodes": [],
            "observed_completed_nodes": ["rag_agent"],
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "final_response"
    assert state["supervisor_dispatch_audit"]["status"] == "ok"
    assert state["supervisor_dispatch_warnings"] == []


def test_audit_supervisor_dispatch_pure_function_does_not_mutate_state():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"]},
        "execution_plan": {"tasks": [{"agent": "rag_agent"}]},
        "completed_nodes": [],
        "status": "completed",
        "approval_payload": {"approval_id": "a1"},
        "supervisor_decision": {"next_expected_node": "rag_agent"},
    }
    before = copy.deepcopy(state)

    audit = audit_supervisor_dispatch(state, "rag_agent")

    assert state == before
    assert audit["supervisor_dispatch_audit"]["status"] == "ok"


def test_graph_wiring_dispatches_after_supervisor_observer():
    graph_path = _ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_builder.py"
    text = graph_path.read_text(encoding="utf-8")

    assert 'workflow.add_edge("supervisor_observer", "llm_supervisor_route")' in text
    assert 'workflow.add_conditional_edges(\n        "llm_supervisor_route"' in text
