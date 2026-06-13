import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.replanner import build_replanner_shadow_report
from src.web_app.tests.db_test_utils import make_test_session


def _patch_final_side_effects(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))


def _final_state(answer="shadow answer"):
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


def test_no_warning_or_failure_does_not_recommend_replan():
    result = build_replanner_shadow_report({
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "agent_results": [],
        "runtime_contract_warnings": [],
    })

    report = result["replanner_shadow_report"]
    assert report["mode"] == "shadow_only"
    assert report["replan_recommended"] is False
    assert report["confidence"] == 0.0
    assert report["suggested_actions"] == ["no_replan_needed"]


def test_supervisor_hint_generates_shadow_recommendation():
    result = build_replanner_shadow_report({
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "supervisor_decision": {
            "should_replan_hint": True,
            "replan_reasons": ["agent_failed:rag_agent"],
        },
    })

    report = result["replanner_shadow_report"]
    assert report["replan_recommended"] is True
    assert "supervisor_decision" in report["trigger_sources"]
    assert "agent_failed:rag_agent" in report["replan_reasons"]
    assert report["suggested_route"] == ["rag_agent", "final_response"]


def test_contract_warning_generates_shadow_recommendation():
    result = build_replanner_shadow_report({
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    })

    report = result["replanner_shadow_report"]
    assert report["replan_recommended"] is True
    assert "runtime_contract" in report["trigger_sources"]
    assert "missing_node_result:rag_agent" in report["replan_reasons"]


def test_failed_formal_agent_result_generates_shadow_recommendation():
    result = build_replanner_shadow_report({
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "agent_results": [{"agent": "rag_agent", "status": "failed", "confidence": 0.0}],
    })

    report = result["replanner_shadow_report"]
    assert report["replan_recommended"] is True
    assert "agent_results" in report["trigger_sources"]
    assert "agent_failed:rag_agent" in report["replan_reasons"]


@pytest.mark.parametrize(
    "state, expected_blocker",
    [
        (
            {
                "status": "waiting_approval",
                "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
                "agent_results": [{"agent": "tool_agent", "status": "needs_approval"}],
            },
            "waiting_approval",
        ),
        (
            {
                "approval_payload": {"approval_id": "ap1"},
                "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
                "runtime_contract_warnings": ["missing_node_result:tool_agent"],
            },
            "approval_pending",
        ),
        (
            {
                "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L4"},
                "runtime_contract_warnings": ["missing_node_result:rag_agent"],
            },
            "risk_level:L4",
        ),
        (
            {
                "route_plan": {"intent": "artifact", "route": ["artifact_agent"], "risk_level": "L1"},
                "agent_results": [{"agent": "artifact_agent", "status": "failed"}],
            },
            "unsafe_intent:artifact",
        ),
        (
            {
                "route_plan": {"intent": "research", "route": ["research_agent"], "risk_level": "L1"},
                "agent_results": [{"agent": "research_agent", "status": "failed"}],
            },
            "unsafe_intent:research",
        ),
    ],
)
def test_unsafe_or_approval_states_are_blocked_shadow_only(state, expected_blocker):
    result = build_replanner_shadow_report(state)

    report = result["replanner_shadow_report"]
    assert report["replan_recommended"] is False
    assert expected_blocker in report["blocked_reasons"]
    assert result["replanner_shadow_warnings"] == ["replanner_shadow_blocked"]


def test_suggested_route_does_not_write_back_route_plan():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }
    original_route_plan = dict(state["route_plan"])

    result = build_replanner_shadow_report(state)

    assert result["replanner_shadow_report"]["suggested_route"] == ["rag_agent", "final_response"]
    assert state["route_plan"] == original_route_plan


@pytest.mark.asyncio
async def test_final_response_payload_contains_replanner_shadow_without_changing_answer(monkeypatch):
    _patch_final_side_effects(monkeypatch)
    state = _final_state()
    state["runtime_contract_warnings"] = ["missing_node_result:rag_agent"]

    result = await RuntimeNodes(make_test_session(), {}).final_response(state)

    assert result["final_answer"] == "shadow answer"
    assert result["final_payload"]["answer"] == "shadow answer"
    assert "replanner_shadow_report" in result
    assert "replanner_shadow_report" in result["final_payload"]
    assert result["final_payload"]["replanner_shadow_report"]["mode"] == "shadow_only"
    assert "node_results" not in result["final_payload"]


def test_missing_runtime_fields_are_safe_defaults():
    result = build_replanner_shadow_report({})

    report = result["replanner_shadow_report"]
    assert report["replan_recommended"] is False
    assert report["current_route"] == []
    assert report["suggested_route"] == []
    assert result["replanner_shadow_warnings"] == []
