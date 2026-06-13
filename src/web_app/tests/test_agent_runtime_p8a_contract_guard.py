import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.contracts import build_runtime_contract_report
from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.tests.db_test_utils import make_test_session


def _patch_final_side_effects(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))


def _final_state(answer="contract answer"):
    return {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route": "chat",
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "completed_nodes": [],
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "final_output": answer,
        "visible_thoughts": [],
        "langgraphstatus": {},
    }


@pytest.mark.asyncio
async def test_final_response_generates_contract_report_without_changing_answer(monkeypatch):
    _patch_final_side_effects(monkeypatch)

    result = await RuntimeNodes(make_test_session(), {}).final_response(_final_state())

    assert result["final_answer"] == "contract answer"
    assert result["final_payload"]["answer"] == "contract answer"
    assert "runtime_contract_report" in result
    assert "runtime_contract_report" in result["final_payload"]
    assert result["final_payload"]["runtime_contract_report"]["node_result_coverage"]["coverage_ok"] is True
    assert "node_results" not in result["final_payload"]
    assert result["node_results"][-1]["delta"]["updates"]["final_payload"] == result["final_payload"]


def test_rag_agent_result_and_node_result_align():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": ["rag_agent"],
        "agent_results": [{"agent": "rag_agent", "status": "ok", "confidence": 0.8}],
        "node_results": [{
            "node": "rag_agent",
            "status": "ok",
            "delta": {"updates": {"rag_result": {"answer": "rag"}}},
        }],
    }

    result = build_runtime_contract_report(state)
    report = result["runtime_contract_report"]

    assert report["node_result_coverage"]["coverage_ok"] is True
    assert report["completion_consistency"]["ok"] is True
    assert report["agent_result_consistency"]["ok"] is True
    assert "agent_node_status_mismatch:rag_agent" not in result["runtime_contract_warnings"]


def test_waiting_approval_tool_node_result_is_valid_without_completion():
    state = {
        "status": "waiting_approval",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
        "completed_nodes": [],
        "agent_results": [{"agent": "tool_agent", "status": "needs_approval", "confidence": 0.8}],
        "node_results": [{
            "node": "tool_agent",
            "status": "needs_approval",
            "delta": {"updates": {"approval_payload": {"approval_id": "ap1"}}},
        }],
    }

    result = build_runtime_contract_report(state)
    report = result["runtime_contract_report"]

    assert report["node_result_coverage"]["coverage_ok"] is True
    assert report["completion_consistency"]["waiting_approval_tool_exception"] is True
    assert report["completion_consistency"]["route_nodes_not_completed"] == []
    assert "route_node_not_completed:tool_agent" not in result["runtime_contract_warnings"]


def test_missing_route_node_result_warns_without_mutating_state():
    state = {
        "status": "running",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "agent_results": [{"agent": "rag_agent", "status": "ok", "confidence": 0.8}],
        "node_results": [],
    }
    before = dict(state)

    result = build_runtime_contract_report(state)

    assert "missing_node_result:rag_agent" in result["runtime_contract_warnings"]
    assert "route_node_not_completed:rag_agent" in result["runtime_contract_warnings"]
    assert "agent_result_without_node_result:rag_agent" in result["runtime_contract_warnings"]
    assert state == before


def test_contract_helper_handles_missing_runtime_fields():
    result = build_runtime_contract_report({})

    assert result["runtime_contract_report"]["node_result_coverage"]["expected_nodes"] == []
    assert result["runtime_contract_report"]["agent_result_consistency"]["ok"] is True
    assert isinstance(result["runtime_contract_warnings"], list)


def test_manifest_extra_update_fields_warn_only():
    state = {
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "node_results": [{
            "node": "final_response",
            "status": "ok",
            "delta": {"updates": {"runtime_latency_trace": {}}},
        }],
    }

    result = build_runtime_contract_report(state)

    assert "manifest_write_contract_extra:final_response" in result["runtime_contract_warnings"]
    assert result["runtime_contract_report"]["manifest_consistency"]["ok"] is False
