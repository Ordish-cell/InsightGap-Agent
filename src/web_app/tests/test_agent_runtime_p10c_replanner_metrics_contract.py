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
from src.web_app.agent.runtime.replanner import (
    build_replanner_candidate_plan,
    build_replanner_control_decision,
    build_replanner_shadow_report,
    update_replanner_shadow_metrics,
)
from src.web_app.tests.db_test_utils import make_test_session


def _patch_settings(monkeypatch, replanner_control_enabled=False):
    settings = SimpleNamespace(agent_replanner_control_enabled=replanner_control_enabled)
    import src.web_app.agent.runtime.replanner as replanner_module

    monkeypatch.setattr(replanner_module, "get_settings", lambda: settings)
    return settings


def _patch_final_side_effects(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))


def _observation(state, legacy_next_node="rag_agent"):
    shadow = build_replanner_shadow_report(state)
    state.update(shadow)
    candidate = build_replanner_candidate_plan(state)
    state.update(candidate)
    control = build_replanner_control_decision(state, legacy_next_node)
    return {**shadow, **candidate, **control}


def test_metrics_accumulate_eligible_blocked_applied_and_fallback(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    observation = _observation(state)
    metrics = update_replanner_shadow_metrics(state, observation)
    state["replanner_shadow_metrics"] = metrics

    assert metrics["shadow_observation_count"] == 1
    assert metrics["candidate_eligible_count"] == 1
    assert metrics["candidate_blocked_count"] == 0
    assert metrics["control_applied_count"] == 1
    assert metrics["control_fallback_count"] == 0
    assert metrics["latest_selected_node"] == "rag_agent"
    assert metrics["latest_legacy_node"] == "rag_agent"
    assert metrics["latest_candidate_node"] == "rag_agent"

    blocked_state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L3"},
        "completed_nodes": [],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
        "replanner_shadow_metrics": metrics,
    }
    blocked_observation = _observation(blocked_state)
    blocked_metrics = update_replanner_shadow_metrics(blocked_state, blocked_observation)

    assert blocked_metrics["shadow_observation_count"] == 2
    assert blocked_metrics["candidate_eligible_count"] == 1
    assert blocked_metrics["candidate_blocked_count"] == 1
    assert blocked_metrics["control_applied_count"] == 1
    assert blocked_metrics["control_fallback_count"] == 1
    assert blocked_metrics["warning_count"] >= 1
    assert blocked_metrics["blocker_counts"]["risk_level:L3"] >= 1


def test_contract_report_checks_replanner_consistency(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }
    observation = _observation(state)
    state.update(observation)
    state["replanner_shadow_metrics"] = update_replanner_shadow_metrics(state, observation)

    result = build_runtime_contract_report(state)
    consistency = result["runtime_contract_report"]["replanner_consistency"]

    assert consistency["shadow_report_present"] is True
    assert consistency["candidate_plan_present"] is True
    assert consistency["candidate_route_from_shadow"] is True
    assert consistency["control_selected_node_allowed"] is True
    assert consistency["control_applied_low_risk"] is True


def test_contract_warns_on_invalid_replanner_control_node():
    result = build_runtime_contract_report({
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "replanner_shadow_report": {
            "suggested_route": ["rag_agent", "final_response"],
            "safety_level": "L1",
        },
        "replanner_candidate_plan": {
            "candidate_route": ["rag_agent", "final_response"],
            "eligible": True,
        },
        "replanner_control_decision": {
            "control_applied": True,
            "selected_next_node": "tool_agent",
        },
    })

    assert "replanner_control_selected_node_not_allowed:tool_agent" in result["runtime_contract_warnings"]
    assert result["runtime_contract_report"]["replanner_consistency"]["ok"] is False


@pytest.mark.asyncio
async def test_final_payload_contains_replanner_metrics_and_control_without_changing_answer(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=False)
    _patch_final_side_effects(monkeypatch)
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route": "rag",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "final_output": "metrics answer",
        "visible_thoughts": [],
        "langgraphstatus": {},
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    result = await RuntimeNodes(make_test_session(), {}).final_response(state)

    assert result["final_answer"] == "metrics answer"
    assert result["final_payload"]["answer"] == "metrics answer"
    assert "replanner_shadow_metrics" in result["final_payload"]
    assert "replanner_control_decision" in result["final_payload"]
    assert "replanner_control_warnings" in result["final_payload"]
    assert result["final_payload"]["replanner_control_decision"]["control_applied"] is False
