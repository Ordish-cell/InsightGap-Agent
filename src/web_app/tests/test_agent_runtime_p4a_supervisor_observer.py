import copy
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.agent.runtime.supervisor import observe_supervisor_state
from src.web_app.tests.db_test_utils import make_test_session


def test_observe_supervisor_state_reports_current_route_without_mutating_plans():
    route_plan = {
        "intent": "rag",
        "route": ["rag_agent", "evaluator", "final_response"],
        "risk_level": "L1",
    }
    execution_plan = execution_plan_from_route_plan(route_plan)
    state = {
        "route_plan": route_plan,
        "execution_plan": execution_plan,
        "completed_nodes": ["rag_agent"],
        "prefetch_results": {"rag": {"evidence": [{"id": 1}]}},
        "parallel_read_results": {
            "rag_prepare": {"status": "ok", "evidence": [{"id": "prepared"}]},
        },
    }
    before = copy.deepcopy(state)

    observation = observe_supervisor_state(state)
    decision = observation["supervisor_decision"]

    assert state == before
    assert decision["mode"] == "observe_only"
    assert decision["current_intent"] == "rag"
    assert decision["current_route"] == ["rag_agent", "evaluator", "final_response"]
    assert decision["next_expected_node"] == "evaluator"
    assert decision["observed_completed_nodes"] == ["rag_agent"]
    assert decision["observed_pending_nodes"] == ["evaluator", "final_response"]
    assert decision["has_prefetch_context"] is True
    assert decision["has_parallel_read_context"] is True
    assert decision["has_rag_prepare_evidence"] is True
    assert decision["rag_prepare_evidence_count"] == 1
    assert "should_replan" not in decision


@pytest.mark.asyncio
async def test_supervisor_observer_writes_only_supervisor_fields():
    db = make_test_session()
    route_plan = {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"}
    execution_plan = execution_plan_from_route_plan(route_plan)
    approval_payload = {"approval_id": "a1"}
    state = {
        "route_plan": route_plan,
        "execution_plan": execution_plan,
        "completed_nodes": [],
        "status": "waiting_approval",
        "approval_payload": approval_payload,
        "pending_tool_call_id": 7,
        "pending_tool_name": "email.send",
        "pending_tool_args": {"to": "a@example.com"},
        "final_payload": {"answer": "paused"},
        "final_output": "paused",
    }
    protected = {
        key: copy.deepcopy(state.get(key))
        for key in (
            "route_plan",
            "execution_plan",
            "completed_nodes",
            "status",
            "approval_payload",
            "pending_tool_call_id",
            "pending_tool_name",
            "pending_tool_args",
            "final_payload",
            "final_output",
        )
    }

    result = await RuntimeNodes(db, {}).supervisor_observer(state)

    for key, value in protected.items():
        assert result.get(key) == value
    assert result["supervisor_decision"]["waiting_approval"] is True
    assert result["supervisor_decision"]["next_expected_node"] == "tool_agent"
    assert "approval_pending" in result["supervisor_warnings"]


def test_supervisor_observer_handles_missing_prefetch_parallel_read_and_agent_results():
    observation = observe_supervisor_state({
        "route_plan": {"intent": "chat", "route": []},
    })
    decision = observation["supervisor_decision"]

    assert decision["current_intent"] == "chat"
    assert decision["current_route"] == []
    assert decision["next_expected_node"] is None
    assert decision["has_prefetch_context"] is False
    assert decision["has_parallel_read_context"] is False
    assert decision["has_rag_prepare_evidence"] is False
    assert decision["rag_prepare_evidence_count"] == 0
    assert decision["failed_agents"] == []
    assert decision["should_replan_hint"] is False


def test_failed_formal_agent_result_creates_replan_hint_without_completing_agent():
    state = {
        "route_plan": {"intent": "artifact", "route": ["artifact_agent", "final_response"]},
        "completed_nodes": [],
        "agent_results": [
            {"agent": "rag_prefetch", "status": "failed", "metadata": {"source": "prefetch"}},
            {"agent": "artifact_agent", "status": "failed", "errors": ["write failed"]},
        ],
    }

    observation = observe_supervisor_state(state)
    decision = observation["supervisor_decision"]

    assert state["completed_nodes"] == []
    assert decision["failed_agents"] == ["artifact_agent"]
    assert decision["should_replan_hint"] is True
    assert "agent_failed:artifact_agent" in decision["replan_reasons"]
    assert "agent_failed:artifact_agent" in observation["supervisor_warnings"]
    assert "rag_prefetch" not in decision["failed_agents"]


def test_graph_wiring_runs_supervisor_between_parallel_read_and_dispatch():
    graph_path = _ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_builder.py"
    text = graph_path.read_text(encoding="utf-8")

    assert 'workflow.add_edge("parallel_read_stage", "supervisor_observer")' in text
    assert 'workflow.add_edge("supervisor_observer", "llm_supervisor_route")' in text
    assert 'workflow.add_conditional_edges(\n        "llm_supervisor_route"' in text
    assert 'workflow.add_conditional_edges(\n            "parallel_read_stage"' not in text
