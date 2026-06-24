import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.dispatch import dispatch_after_evaluator
from src.web_app.agent.runtime.recovery import (
    apply_evaluator_recovery_decision,
    build_evaluator_recovery_decision,
)
from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.tests.db_test_utils import make_test_session


def _runtime() -> AgentRuntime:
    return AgentRuntime(make_test_session(), {})


def _state(route, *, warnings=None, agent_results=None, risk_level="L1", **extra):
    return {
        "route_plan": {"intent": "rag", "route": route, "risk_level": risk_level},
        "completed_nodes": [],
        "agent_results": agent_results or [],
        "final_warnings": warnings or [],
        **extra,
    }


def test_rag_evidence_missing_retries_rag_agent():
    state = _state(["rag_agent", "evaluator", "final_response"], warnings=["rag_evidence_missing"])

    decision = build_evaluator_recovery_decision(state, state["final_warnings"])
    apply_evaluator_recovery_decision(state, decision)

    assert decision["should_retry"] is True
    assert decision["target"] == "rag_agent"
    assert state["evaluator_recovery_attempts"] == {"rag_agent": 1}
    assert dispatch_after_evaluator(state) == "rag_agent"


def test_route_order_selects_first_recoverable_failure():
    state = _state(
        ["artifact_agent", "tool_agent", "evaluator", "final_response"],
        warnings=["tool_failed", "artifact_missing"],
    )

    decision = build_evaluator_recovery_decision(state, state["final_warnings"])

    assert decision["target"] == "artifact_agent"


def test_tool_artifact_memory_skill_and_research_failures_are_recoverable():
    cases = [
        ("tool_agent", "tool_failed"),
        ("artifact_agent", "artifact_missing"),
        ("memory_agent", "memory_write_failed"),
        ("skill_agent", "skill_agent_failed"),
        ("research_agent", "research_failed"),
    ]

    for agent, warning in cases:
        state = _state([agent, "evaluator", "final_response"], warnings=[warning])
        decision = build_evaluator_recovery_decision(state, state["final_warnings"])

        assert decision["should_retry"] is True
        assert decision["target"] == agent


def test_denied_waiting_approval_and_l4_do_not_retry():
    cases = [
        _state(["tool_agent"], warnings=["tool_denied"]),
        _state(["tool_agent"], warnings=["tool_failed"], status="waiting_approval"),
        _state(["tool_agent"], warnings=["tool_failed"], approval_payload={"id": "a1"}),
        _state(["tool_agent"], warnings=["tool_failed"], risk_level="L4"),
    ]

    for state in cases:
        decision = build_evaluator_recovery_decision(state, state["final_warnings"])

        assert decision["should_retry"] is False
        assert dispatch_after_evaluator({"evaluator_recovery_decision": decision}) == "final_response"


def test_retry_limit_exhausts_to_final_response():
    state = _state(
        ["rag_agent", "evaluator", "final_response"],
        warnings=["rag_evidence_missing"],
        evaluator_recovery_attempts={"rag_agent": 1},
    )

    decision = build_evaluator_recovery_decision(state, state["final_warnings"])
    apply_evaluator_recovery_decision(state, decision)

    assert decision["should_retry"] is False
    assert decision["exhausted"] is True
    assert state["evaluator_recovery_active"] is False
    assert dispatch_after_evaluator(state) == "final_response"


def test_old_failed_agent_result_does_not_retry_after_latest_success():
    state = _state(
        ["rag_agent", "evaluator", "final_response"],
        agent_results=[
            {"agent": "rag_agent", "status": "failed"},
            {"agent": "rag_agent", "status": "ok", "evidence": [{"id": "e1"}]},
        ],
    )

    decision = build_evaluator_recovery_decision(state, [])

    assert decision["should_retry"] is False
    assert decision["target"] == ""


def test_recovery_agent_returns_to_evaluator_before_final_response():
    state = _state(
        ["rag_agent", "evaluator", "final_response"],
        evaluator_recovery_active=True,
        evaluator_recovery_target="rag_agent",
        completed_nodes=["rag_agent"],
    )

    assert _runtime()._dispatch_next_route_node(state) == "evaluator"
