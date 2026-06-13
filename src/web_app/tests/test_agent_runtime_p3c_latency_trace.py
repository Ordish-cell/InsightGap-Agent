import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import nodes as runtime_nodes
from src.web_app.agent.runtime.latency import build_runtime_latency_trace
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.tests.db_test_utils import make_test_session


def test_latency_trace_summarizes_prefetch_parallel_read_and_rag_prepare():
    state = {
        "prefetch_elapsed_ms": 31,
        "prefetch_results": {
            "rag": {"evidence": [{"id": "p1"}]},
            "memory": {"items": [{"id": "m1"}, {"id": "m2"}]},
            "skill": {"matched_skill": {"name": "s"}},
        },
        "parallel_read_elapsed_ms": 42,
        "parallel_read_branch_timings": {"context_skill": 40, "rag_prepare": 12},
        "parallel_read_results": {
            "rag_prepare": {
                "status": "ok",
                "elapsed_ms": 12,
                "from_prefetch": True,
                "search_attempted": True,
                "evidence_count": 1,
                "evidence": [{"id": "r1"}],
            }
        },
        "supervisor_dispatch_audit": {"status": "ok", "legacy_next_node": "rag_agent"},
        "supervisor_readiness_report": {"readiness_level": "eligible_candidate", "ready_for_control": True},
        "supervisor_control_decision": {"control_applied": True, "selected_next_node": "rag_agent", "fallback_reason": ""},
        "agent_results": [{"agent": "rag_agent", "status": "ok"}],
    }

    result = build_runtime_latency_trace(state, elapsed_ms=100)
    trace = result["runtime_latency_trace"]

    assert trace["total_elapsed_ms"] == 100
    assert trace["prefetch"]["elapsed_ms"] == 31
    assert trace["prefetch"]["rag_evidence_count"] == 1
    assert trace["prefetch"]["memory_count"] == 2
    assert trace["prefetch"]["skill_matched"] is True
    assert trace["parallel_read"]["elapsed_ms"] == 42
    assert trace["parallel_read"]["branch_timings"]["rag_prepare"] == 12
    assert trace["parallel_read"]["rag_prepare"]["evidence_count"] == 1
    assert trace["parallel_read"]["rag_prepare"]["no_evidence"] is False
    assert trace["supervisor"]["control_applied"] is True
    assert trace["agent_results"]["count"] == 1
    assert result["runtime_latency_warnings"] == []


def test_latency_trace_marks_rag_prepare_no_evidence():
    result = build_runtime_latency_trace({
        "parallel_read_results": {
            "rag_prepare": {
                "status": "ok",
                "elapsed_ms": 9,
                "search_attempted": True,
                "evidence_count": 0,
                "evidence": [],
            }
        }
    })

    trace = result["runtime_latency_trace"]
    assert trace["parallel_read"]["rag_prepare"]["search_attempted"] is True
    assert trace["parallel_read"]["rag_prepare"]["evidence_count"] == 0
    assert trace["parallel_read"]["rag_prepare"]["no_evidence"] is True


def test_latency_trace_records_supervisor_control_fallback_without_dispatch_change():
    result = build_runtime_latency_trace({
        "supervisor_dispatch_audit": {"status": "mismatch", "legacy_next_node": "rag_agent"},
        "supervisor_readiness_report": {"readiness_level": "blocked", "ready_for_control": False},
        "supervisor_control_decision": {
            "control_applied": False,
            "selected_next_node": "rag_agent",
            "fallback_reason": "dispatch_mismatch",
        },
        "supervisor_control_warnings": ["supervisor_control_fallback:dispatch_mismatch"],
    })

    trace = result["runtime_latency_trace"]
    assert trace["supervisor"]["dispatch_status"] == "mismatch"
    assert trace["supervisor"]["control_applied"] is False
    assert trace["supervisor"]["selected_next_node"] == "rag_agent"
    assert trace["supervisor"]["fallback_reason"] == "dispatch_mismatch"
    assert "supervisor_warnings_present" in result["runtime_latency_warnings"]


def test_latency_trace_defaults_when_optional_runtime_fields_are_missing():
    result = build_runtime_latency_trace({})
    trace = result["runtime_latency_trace"]

    assert trace["mode"] == "runtime_latency_trace"
    assert trace["total_elapsed_ms"] is None
    assert trace["prefetch"]["elapsed_ms"] == 0
    assert trace["parallel_read"]["elapsed_ms"] == 0
    assert trace["parallel_read"]["rag_prepare"]["status"] == ""
    assert trace["supervisor"]["control_applied"] is False
    assert trace["agent_results"]["count"] == 0


@pytest.mark.asyncio
async def test_final_response_payload_includes_latency_trace_without_changing_answer(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route_plan": {"intent": "chat", "route": ["final_response"], "risk_level": "L0"},
        "execution_plan": execution_plan_from_route_plan({"intent": "chat", "route": ["final_response"]}),
        "final_output": "final answer",
        "prefetch_elapsed_ms": 5,
        "parallel_read_elapsed_ms": 7,
        "parallel_read_results": {
            "rag_prepare": {
                "status": "ok",
                "search_attempted": True,
                "evidence_count": 0,
                "evidence": [],
            }
        },
    }

    result = await RuntimeNodes(db, {}).final_response(state)

    assert result["final_answer"] == "final answer"
    trace = result["final_payload"]["runtime_latency_trace"]
    assert trace["prefetch"]["elapsed_ms"] == 5
    assert trace["parallel_read"]["elapsed_ms"] == 7
    assert trace["parallel_read"]["rag_prepare"]["no_evidence"] is True
    assert result["runtime_latency_trace"] == trace
