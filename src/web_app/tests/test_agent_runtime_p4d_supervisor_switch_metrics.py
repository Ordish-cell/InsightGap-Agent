import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.supervisor import (
    audit_supervisor_dispatch,
    build_supervisor_shadow_policy,
    update_supervisor_shadow_metrics,
)
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


@pytest.mark.asyncio
async def test_supervisor_master_switch_disables_observer_and_audit(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_enabled=False)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
    }

    observed = await RuntimeNodes(make_test_session(), {}).supervisor_observer(state)
    next_node = _runtime()._dispatch_next_route_node(observed)

    assert next_node == "rag_agent"
    assert observed["supervisor_decision"] == {}
    assert "supervisor_disabled" in observed["supervisor_warnings"]
    assert observed["supervisor_dispatch_audit"]["status"] == "skipped"
    assert observed["supervisor_dispatch_audit"]["reason"] == "supervisor_disabled"
    assert observed["supervisor_shadow_policy"]["mode"] == "disabled"
    assert "supervisor_disabled" in observed["supervisor_shadow_policy"]["control_blockers"]
    assert observed["supervisor_shadow_metrics"]["supervisor_enabled"] is False


def test_shadow_policy_switch_disables_policy_without_changing_legacy_dispatch(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_shadow_policy_enabled=False)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "supervisor_decision": {"next_expected_node": "rag_agent"},
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["supervisor_shadow_policy"]["mode"] == "disabled"
    assert state["supervisor_shadow_policy"]["control_eligible"] is False
    assert "shadow_policy_disabled" in state["supervisor_shadow_policy"]["control_blockers"]
    assert state["supervisor_shadow_metrics"]["shadow_policy_enabled"] is False


def test_metrics_switch_disables_metrics_payload(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_shadow_metrics_enabled=False)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "supervisor_decision": {"next_expected_node": "rag_agent"},
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["supervisor_shadow_metrics"] == {"enabled": False}


def test_shadow_metrics_accumulate_dispatch_counts(monkeypatch):
    _patch_settings(monkeypatch)
    state = {
        "route_plan": {"intent": "artifact", "route": ["artifact_agent"], "risk_level": "L2"},
        "completed_nodes": [],
        "supervisor_decision": {"next_expected_node": "evaluator"},
    }

    first = audit_supervisor_dispatch(state, "artifact_agent")
    state["supervisor_shadow_metrics"] = update_supervisor_shadow_metrics(state, first)
    second = audit_supervisor_dispatch(state, "artifact_agent")
    state["supervisor_shadow_metrics"] = update_supervisor_shadow_metrics(state, second)

    metrics = state["supervisor_shadow_metrics"]
    assert metrics["enabled"] is True
    assert metrics["dispatch_audit_count"] == 2
    assert metrics["dispatch_mismatch_count"] == 2
    assert metrics["control_blocked_count"] == 2
    assert metrics["warning_count"] >= 2
    assert metrics["blocker_counts"]["unsafe_intent:artifact"] == 2
    assert metrics["blocker_counts"]["unsafe_agent:artifact_agent"] == 2


def test_control_flag_is_recorded_but_does_not_override_legacy_dispatch(monkeypatch):
    _patch_settings(monkeypatch, agent_supervisor_control_enabled=True)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "supervisor_decision": {"next_expected_node": "evaluator"},
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["supervisor_shadow_policy"]["control_enabled"] is True
    assert state["supervisor_shadow_policy"]["recommended_next_node"] == "evaluator"
    assert state["supervisor_shadow_policy"]["legacy_next_node"] == "rag_agent"


def test_build_policy_defaults_control_enabled_to_false(monkeypatch):
    _patch_settings(monkeypatch)
    policy = build_supervisor_shadow_policy({
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "supervisor_decision": {"next_expected_node": "rag_agent"},
    }, "rag_agent")

    assert policy["mode"] == "shadow_only"
    assert policy["control_enabled"] is False
