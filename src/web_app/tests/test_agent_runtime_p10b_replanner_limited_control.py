import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.tests.db_test_utils import make_test_session


def _runtime() -> AgentRuntime:
    return AgentRuntime(make_test_session(), {})


def _patch_settings(monkeypatch, replanner_control_enabled=False):
    settings = SimpleNamespace(
        agent_supervisor_enabled=True,
        agent_supervisor_shadow_policy_enabled=True,
        agent_supervisor_shadow_metrics_enabled=True,
        agent_supervisor_control_enabled=False,
        agent_replanner_control_enabled=replanner_control_enabled,
    )
    import src.web_app.agent.runtime.replanner as replanner_module
    import src.web_app.agent.runtime.supervisor as supervisor_module

    monkeypatch.setattr(replanner_module, "get_settings", lambda: settings)
    monkeypatch.setattr(supervisor_module, "get_settings", lambda: settings)
    return settings


def test_control_disabled_always_returns_legacy(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=False)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    decision = state["replanner_control_decision"]
    assert decision["control_applied"] is False
    assert decision["selected_next_node"] == "rag_agent"
    assert decision["fallback_reason"] == "control_disabled"


def test_control_enabled_low_risk_rag_candidate_can_return_rag_agent(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    decision = state["replanner_control_decision"]
    assert decision["control_applied"] is True
    assert decision["selected_next_node"] == "rag_agent"
    assert decision["candidate_next_node"] == "rag_agent"


def test_control_enabled_low_risk_chat_candidate_can_return_final_response(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route": "chat",
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "completed_nodes": [],
        "supervisor_decision": {
            "mode": "observe_only",
            "current_intent": "chat",
            "next_expected_node": None,
            "should_replan_hint": True,
            "replan_reasons": ["supervisor_should_replan_hint"],
        },
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "final_response"
    decision = state["replanner_control_decision"]
    assert decision["control_applied"] is True
    assert decision["selected_next_node"] == "final_response"
    assert decision["candidate_next_node"] == "final_response"


def test_completed_agent_is_not_retried(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent", "final_response"], "risk_level": "L1"},
        "completed_nodes": ["rag_agent"],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "final_response"
    assert state["replanner_control_decision"]["selected_next_node"] == "final_response"
    assert state["replanner_control_decision"]["candidate_next_node"] == "final_response"


@pytest.mark.parametrize(
    "extra_state, expected_blocker",
    [
        (
            {
                "supervisor_decision": {
                    "mode": "observe_only",
                    "current_intent": "rag",
                    "next_expected_node": "final_response",
                },
            },
            "supervisor_dispatch_mismatch",
        ),
        ({"approval_payload": {"approval_id": "a1"}}, "approval_pending"),
        ({"route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L3"}}, "risk_level:L3"),
        ({"route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L4"}}, "risk_level:L4"),
    ],
)
def test_mismatch_pending_approval_and_high_risk_fallback_to_legacy(monkeypatch, extra_state, expected_blocker):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
        **extra_state,
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    decision = state["replanner_control_decision"]
    assert decision["control_applied"] is False
    assert decision["selected_next_node"] == "rag_agent"
    assert expected_blocker in decision["blockers"]


@pytest.mark.parametrize(
    "intent, route, expected_next, expected_blocker",
    [
        ("tool.email", ["tool_agent"], "tool_agent", "unsafe_intent:tool.email"),
        ("artifact", ["artifact_agent"], "artifact_agent", "unsafe_intent:artifact"),
        ("memory", ["memory_agent"], "memory_agent", "unsafe_intent:memory"),
        ("skill", ["skill_agent"], "skill_agent", "unsafe_intent:skill"),
        ("research", ["research_agent"], "research_agent", "unsafe_intent:research"),
    ],
)
def test_tool_artifact_memory_skill_and_research_routes_fallback_to_legacy(
    monkeypatch,
    intent,
    route,
    expected_next,
    expected_blocker,
):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route_plan": {"intent": intent, "route": route, "risk_level": "L1"},
        "completed_nodes": [],
        "runtime_contract_warnings": [f"missing_node_result:{route[0]}"],
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == expected_next
    decision = state["replanner_control_decision"]
    assert decision["control_applied"] is False
    assert expected_blocker in decision["blockers"]


def test_replanner_control_does_not_mutate_protected_fields(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": {"tasks": [{"agent": "rag_agent"}]},
        "completed_nodes": [],
        "status": "running",
        "approval_payload": None,
        "pending_tool_call_id": None,
        "pending_approval_id": None,
        "pending_tool_name": None,
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
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

    _runtime()._dispatch_next_route_node(state)

    for key, value in before.items():
        assert state.get(key) == value
