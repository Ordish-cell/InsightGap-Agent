import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.replanner import build_replanner_candidate_plan
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.tests.db_test_utils import make_test_session


def _patch_final_side_effects(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))


def _shadow_route(recommended=True, blocked=None, route=None):
    route = route or ["rag_agent"]
    return {
        "mode": "shadow_only",
        "replan_recommended": recommended,
        "confidence": 0.65 if recommended else 0.0,
        "trigger_sources": ["runtime_contract"] if recommended else [],
        "replan_reasons": ["missing_node_result:rag_agent"] if recommended else [],
        "blocked_reasons": blocked or [],
        "current_route": route,
        "suggested_route": [*route, "final_response"] if recommended and "final_response" not in route else route,
        "suggested_actions": ["shadow_replan_candidate"] if recommended else ["no_replan_needed"],
        "safety_level": "L1",
    }


def test_low_risk_rag_shadow_recommendation_generates_eligible_candidate():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": {"plan_id": "original", "tasks": [{"agent": "rag_agent"}]},
        "replanner_shadow_report": _shadow_route(),
    }

    result = build_replanner_candidate_plan(state)
    candidate = result["replanner_candidate_plan"]

    assert candidate["mode"] == "candidate_only"
    assert candidate["source"] == "replanner_shadow"
    assert candidate["eligible"] is True
    assert candidate["current_route"] == ["rag_agent"]
    assert candidate["candidate_route"] == ["rag_agent", "final_response"]
    assert [task["agent"] for task in candidate["candidate_execution_plan"]["tasks"]] == ["rag_agent", "final_response"]
    assert candidate["candidate_execution_plan"] is not state["execution_plan"]
    assert state["execution_plan"]["plan_id"] == "original"


def test_no_recommendation_is_not_eligible():
    state = {
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "replanner_shadow_report": _shadow_route(recommended=False, route=[]),
    }

    candidate = build_replanner_candidate_plan(state)["replanner_candidate_plan"]

    assert candidate["eligible"] is False
    assert "shadow_not_recommended" in candidate["blocked_reasons"]


def test_blocked_shadow_report_is_not_eligible_and_preserves_blockers():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "replanner_shadow_report": _shadow_route(blocked=["pending_tool_or_approval"]),
    }

    candidate = build_replanner_candidate_plan(state)["replanner_candidate_plan"]

    assert candidate["eligible"] is False
    assert "pending_tool_or_approval" in candidate["blocked_reasons"]


@pytest.mark.parametrize(
    "intent, route, blocker",
    [
        ("tool.email", ["tool_agent"], "unsafe_intent:tool.email"),
        ("artifact", ["artifact_agent"], "unsafe_intent:artifact"),
        ("memory", ["memory_agent"], "unsafe_intent:memory"),
        ("skill", ["skill_agent"], "unsafe_intent:skill"),
        ("research", ["research_agent"], "unsafe_intent:research"),
    ],
)
def test_unsafe_routes_do_not_generate_eligible_candidate(intent, route, blocker):
    state = {
        "route_plan": {"intent": intent, "route": route, "risk_level": "L1"},
        "replanner_shadow_report": _shadow_route(route=route),
    }

    candidate = build_replanner_candidate_plan(state)["replanner_candidate_plan"]

    assert candidate["eligible"] is False
    assert blocker in candidate["blocked_reasons"]


@pytest.mark.parametrize(
    "extra_state, blocker",
    [
        ({"route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L3"}}, "risk_level:L3"),
        ({"status": "waiting_approval"}, "waiting_approval"),
        ({"pending_tool_call_id": "tc1"}, "pending_tool_or_approval"),
        ({"approval_payload": {"approval_id": "ap1"}}, "approval_pending"),
    ],
)
def test_high_risk_or_pending_state_is_not_eligible(extra_state, blocker):
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "replanner_shadow_report": _shadow_route(),
        **extra_state,
    }

    candidate = build_replanner_candidate_plan(state)["replanner_candidate_plan"]

    assert candidate["eligible"] is False
    assert blocker in candidate["blocked_reasons"]


def test_candidate_does_not_mutate_route_plan_or_execution_plan():
    route_plan = {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"}
    execution_plan = execution_plan_from_route_plan(route_plan)
    state = {
        "route_plan": route_plan,
        "execution_plan": execution_plan,
        "replanner_shadow_report": _shadow_route(),
    }
    original_route_plan = dict(route_plan)
    original_execution_plan = dict(execution_plan)

    result = build_replanner_candidate_plan(state)

    assert result["replanner_candidate_plan"]["eligible"] is True
    assert state["route_plan"] == original_route_plan
    assert state["execution_plan"] == original_execution_plan


@pytest.mark.asyncio
async def test_final_response_payload_contains_candidate_without_changing_answer(monkeypatch):
    _patch_final_side_effects(monkeypatch)
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route": "rag",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"}),
        "completed_nodes": [],
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "final_output": "candidate answer",
        "visible_thoughts": [],
        "langgraphstatus": {},
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    result = await RuntimeNodes(make_test_session(), {}).final_response(state)

    assert result["final_answer"] == "candidate answer"
    assert result["final_payload"]["answer"] == "candidate answer"
    assert "replanner_candidate_plan" in result
    assert "replanner_candidate_plan" in result["final_payload"]
    assert result["final_payload"]["replanner_candidate_plan"]["mode"] == "candidate_only"
    assert result["route_plan"]["route"] == ["rag_agent"]
    assert [task["agent"] for task in result["execution_plan"]["tasks"]] == ["rag_agent"]
