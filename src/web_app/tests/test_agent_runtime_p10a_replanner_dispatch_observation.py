import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph import _END_SENTINEL, AgentRuntime
from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
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


def _patch_final_side_effects(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))


def test_dispatch_writes_replanner_observation_but_returns_legacy_next(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=False)
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["replanner_shadow_report"]["mode"] == "shadow_only"
    assert state["replanner_candidate_plan"]["mode"] == "candidate_only"
    assert state["replanner_candidate_plan"]["eligible"] is True
    assert state["replanner_control_decision"]["control_applied"] is False
    assert state["replanner_control_decision"]["selected_next_node"] == "rag_agent"
    assert state["replanner_shadow_metrics"]["shadow_observation_count"] == 1


def test_waiting_approval_still_returns_end(monkeypatch):
    _patch_settings(monkeypatch, replanner_control_enabled=True)
    state = {
        "status": "waiting_approval",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
        "completed_nodes": [],
        "approval_payload": {"approval_id": "a1"},
        "pending_approval_id": "a1",
        "pending_tool_call_id": 1,
        "pending_tool_name": "email.send",
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == _END_SENTINEL
    assert state["status"] == "waiting_approval"
    assert "waiting_approval" in state["replanner_control_decision"]["blockers"]
    assert state["replanner_control_decision"]["selected_next_node"] == _END_SENTINEL


def test_dispatch_replanner_observation_does_not_modify_protected_fields(monkeypatch):
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


@pytest.mark.asyncio
async def test_final_response_retains_replanner_payload(monkeypatch):
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
        "final_output": "replanner answer",
        "visible_thoughts": [],
        "langgraphstatus": {},
        "runtime_contract_warnings": ["missing_node_result:rag_agent"],
    }

    result = await RuntimeNodes(make_test_session(), {}).final_response(state)

    assert result["final_answer"] == "replanner answer"
    assert result["final_payload"]["answer"] == "replanner answer"
    assert "replanner_shadow_report" in result["final_payload"]
    assert "replanner_candidate_plan" in result["final_payload"]
    assert "replanner_control_decision" in result["final_payload"]
    assert "replanner_shadow_metrics" in result["final_payload"]
