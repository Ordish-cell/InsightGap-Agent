import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph import _END_SENTINEL, AgentRuntime
from src.web_app.agent.runtime.supervisor import build_supervisor_readiness_report
from src.web_app.tests.db_test_utils import make_test_session


def _runtime() -> AgentRuntime:
    return AgentRuntime(make_test_session(), {})


def _patch_settings(monkeypatch, **overrides):
    defaults = {
        "agent_supervisor_enabled": True,
        "agent_supervisor_shadow_policy_enabled": True,
        "agent_supervisor_shadow_metrics_enabled": True,
        "agent_supervisor_control_enabled": False,
    }
    defaults.update(overrides)
    settings = SimpleNamespace(**defaults)
    import src.web_app.agent.runtime.supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "get_settings", lambda: settings)
    return settings


@pytest.mark.parametrize(
    ("intent", "route", "expected_next", "risk_level"),
    [
        ("rag", ["rag_agent"], "rag_agent", "L1"),
        ("chat", [], "final_response", "L0"),
    ],
)
def test_low_risk_chat_and_rag_can_be_eligible_candidates(intent, route, expected_next, risk_level):
    state = {
        "route_plan": {"intent": intent, "route": route, "risk_level": risk_level},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": intent,
            "next_expected_node": route[0] if route else None,
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == expected_next
    report = state["supervisor_readiness_report"]
    assert report["mode"] == "readiness_only"
    assert report["ready_for_control"] is True
    assert report["readiness_level"] == "eligible_candidate"
    assert report["recommended_next_phase"] == "p5a_candidate"
    assert report["blockers"] == []


def test_dispatch_mismatch_blocks_readiness_without_changing_dispatch():
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": "rag",
            "next_expected_node": "evaluator",
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    report = state["supervisor_readiness_report"]
    assert report["ready_for_control"] is False
    assert report["readiness_level"] == "blocked"
    assert "dispatch_mismatch" in report["blockers"]
    assert "metrics_dispatch_mismatch" in report["blockers"]


@pytest.mark.parametrize("risk_level", ["L3", "L4"])
def test_waiting_approval_pending_and_high_risk_are_blocked(risk_level):
    approval_payload = {"approval_id": "a1"}
    state = {
        "status": "waiting_approval",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": risk_level},
        "completed_nodes": [],
        "approval_payload": approval_payload,
        "approval_required": True,
        "pending_approval_id": "a1",
        "pending_tool_call_id": 7,
        "pending_tool_name": "email.send",
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": "tool.email",
            "next_expected_node": "tool_agent",
            "waiting_approval": True,
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == _END_SENTINEL
    assert state["status"] == "waiting_approval"
    assert state["approval_payload"] is approval_payload
    report = state["supervisor_readiness_report"]
    assert report["ready_for_control"] is False
    assert report["readiness_level"] == "blocked"
    assert "waiting_approval" in report["blockers"]
    assert "approval_pending" in report["blockers"]
    assert "pending_tool_or_approval" in report["blockers"]
    assert f"risk_level:{risk_level}" in report["blockers"]
    assert "dispatch_skipped:waiting_approval" in report["blockers"]


def test_tool_artifact_memory_skill_and_research_routes_are_blocked():
    cases = [
        ("tool.email", ["tool_agent"], "unsafe_intent:tool.email", "unsafe_agent:tool_agent"),
        ("artifact", ["artifact_agent"], "unsafe_intent:artifact", "unsafe_agent:artifact_agent"),
        ("memory", ["memory_agent"], "unsafe_intent:memory", "unsafe_agent:memory_agent"),
        ("skill", ["skill_agent"], "unsafe_intent:skill", "unsafe_agent:skill_agent"),
        ("research", ["research_agent"], "unsafe_intent:research", "unsafe_agent:research_agent"),
    ]

    for intent, route, intent_blocker, agent_blocker in cases:
        state = {
            "route_plan": {"intent": intent, "route": route, "risk_level": "L1"},
            "completed_nodes": [],
            "supervisor_decision": {
                "mode": "observe_only",
                "current_intent": intent,
                "next_expected_node": route[0],
            },
        }

        next_node = _runtime()._dispatch_next_route_node(state)

        assert next_node == route[0]
        report = state["supervisor_readiness_report"]
        assert report["ready_for_control"] is False
        assert report["readiness_level"] == "blocked"
        assert intent_blocker in report["blockers"]
        assert agent_blocker in report["blockers"]


def test_control_enabled_true_is_observed_but_does_not_change_dispatch(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": "rag",
            "next_expected_node": "rag_agent",
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    report = state["supervisor_readiness_report"]
    assert report["ready_for_control"] is True
    assert report["metrics_summary"]["control_enabled"] is True
    assert "control_flag_observed_but_not_applied" in state["supervisor_readiness_warnings"]


def test_missing_shadow_signals_produces_blocked_report_without_error():
    readiness = build_supervisor_readiness_report({})

    report = readiness["supervisor_readiness_report"]
    assert report["mode"] == "readiness_only"
    assert report["ready_for_control"] is False
    assert report["readiness_level"] == "blocked"
    assert "missing_shadow_metrics" in report["blockers"]
    assert "missing_shadow_policy" in report["blockers"]
    assert "missing_dispatch_audit" in report["blockers"]
