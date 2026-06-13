import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph import _END_SENTINEL, AgentRuntime
from src.web_app.agent.runtime.supervisor import build_supervisor_control_decision
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


def test_control_disabled_returns_legacy_even_when_readiness_is_eligible(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=False)
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
    assert state["supervisor_readiness_report"]["ready_for_control"] is True
    decision = state["supervisor_control_decision"]
    assert decision["control_applied"] is False
    assert decision["selected_next_node"] == "rag_agent"
    assert decision["fallback_reason"] == "control_disabled"


def test_control_enabled_low_risk_rag_applies_limited_control(monkeypatch):
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
    decision = state["supervisor_control_decision"]
    assert decision["mode"] == "limited_control"
    assert decision["control_applied"] is True
    assert decision["selected_next_node"] == "rag_agent"
    assert decision["legacy_next_node"] == "rag_agent"
    assert decision["recommended_next_node"] == "rag_agent"
    assert decision["blockers"] == []


def test_control_enabled_low_risk_chat_applies_final_response_control(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=True)
    state = {
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": "chat",
            "next_expected_node": None,
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "final_response"
    decision = state["supervisor_control_decision"]
    assert decision["control_applied"] is True
    assert decision["selected_next_node"] == "final_response"
    assert decision["legacy_next_node"] == "final_response"
    assert decision["recommended_next_node"] == "final_response"


def test_mismatch_does_not_apply_control_and_returns_legacy(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": "rag",
            "next_expected_node": "final_response",
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    decision = state["supervisor_control_decision"]
    assert decision["control_applied"] is False
    assert decision["selected_next_node"] == "rag_agent"
    assert decision["fallback_reason"]
    assert "dispatch_mismatch" in decision["blockers"]


@pytest.mark.parametrize("risk_level", ["L3", "L4"])
def test_waiting_approval_pending_and_high_risk_never_apply_control(monkeypatch, risk_level):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=True)
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
    decision = state["supervisor_control_decision"]
    assert decision["control_applied"] is False
    assert decision["selected_next_node"] == _END_SENTINEL
    assert "waiting_approval" in decision["blockers"]
    assert "pending_tool_or_approval" in decision["blockers"]
    assert f"risk_level:{risk_level}" in decision["blockers"]


def test_write_and_research_routes_never_apply_control(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=True)
    cases = [
        ("tool.email", ["tool_agent"], "unsafe_agent:tool_agent"),
        ("artifact", ["artifact_agent"], "unsafe_agent:artifact_agent"),
        ("memory", ["memory_agent"], "unsafe_agent:memory_agent"),
        ("skill", ["skill_agent"], "unsafe_agent:skill_agent"),
        ("research", ["research_agent"], "unsafe_agent:research_agent"),
    ]

    for intent, route, expected_blocker in cases:
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
        decision = state["supervisor_control_decision"]
        assert decision["control_applied"] is False
        assert expected_blocker in decision["blockers"]


def test_recommended_node_outside_allowlist_falls_back_to_legacy():
    state = {
        "supervisor_readiness_report": {
            "mode": "readiness_only",
            "ready_for_control": True,
            "readiness_level": "eligible_candidate",
            "blockers": [],
        },
        "supervisor_shadow_policy": {
            "mode": "shadow_only",
            "control_enabled": True,
            "control_eligible": True,
            "recommended_next_node": "evaluator",
            "control_blockers": [],
        },
        "supervisor_dispatch_audit": {
            "status": "ok",
            "legacy_next_node": "final_response",
            "matched": True,
        },
    }

    result = build_supervisor_control_decision(state, "final_response")

    decision = result["supervisor_control_decision"]
    assert decision["control_applied"] is False
    assert decision["selected_next_node"] == "final_response"
    assert "recommended_node_not_allowed:evaluator" in decision["blockers"]


def test_control_decision_does_not_mutate_protected_state_fields(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": {"tasks": [{"agent": "rag_agent"}]},
        "completed_nodes": [],
        "status": "completed",
        "approval_payload": {"approval_id": "a1"},
        "pending_tool_call_id": None,
        "pending_approval_id": None,
        "pending_tool_name": None,
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": "rag",
            "next_expected_node": "rag_agent",
        },
    }
    protected_keys = (
        "route_plan",
        "execution_plan",
        "completed_nodes",
        "status",
        "approval_payload",
        "pending_tool_call_id",
        "pending_approval_id",
        "pending_tool_name",
    )
    before = {key: copy.deepcopy(state.get(key)) for key in protected_keys}

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    for key, value in before.items():
        assert state.get(key) == value
