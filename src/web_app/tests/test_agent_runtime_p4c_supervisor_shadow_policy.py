import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph import _END_SENTINEL, AgentRuntime
from src.web_app.agent.runtime.supervisor import build_supervisor_shadow_policy
from src.web_app.tests.db_test_utils import make_test_session


def _runtime() -> AgentRuntime:
    return AgentRuntime(make_test_session(), {})


def test_low_risk_rag_shadow_policy_can_be_control_eligible_but_does_not_override_dispatch():
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
    assert state["supervisor_shadow_policy"]["mode"] == "shadow_only"
    assert state["supervisor_shadow_policy"]["legacy_next_node"] == "rag_agent"
    assert state["supervisor_shadow_policy"]["recommended_next_node"] == "rag_agent"
    assert state["supervisor_shadow_policy"]["control_eligible"] is True
    assert state["supervisor_shadow_policy"]["control_blockers"] == []


def test_shadow_policy_recommendation_never_overrides_legacy_dispatch():
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
    assert state["supervisor_shadow_policy"]["recommended_next_node"] == "evaluator"
    assert state["supervisor_shadow_policy"]["legacy_next_node"] == "rag_agent"
    assert state["supervisor_dispatch_audit"]["status"] == "mismatch"


def test_tool_artifact_memory_skill_research_routes_are_not_control_eligible():
    cases = [
        ("tool.email", ["tool_agent"], "unsafe_intent:tool.email", "unsafe_agent:tool_agent"),
        ("artifact", ["artifact_agent"], "unsafe_intent:artifact", "unsafe_agent:artifact_agent"),
        ("memory", ["memory_agent"], "unsafe_intent:memory", "unsafe_agent:memory_agent"),
        ("skill", ["skill_agent"], "unsafe_intent:skill", "unsafe_agent:skill_agent"),
        ("research", ["research_agent"], "unsafe_intent:research", "unsafe_agent:research_agent"),
    ]

    for intent, route, intent_blocker, agent_blocker in cases:
        policy = build_supervisor_shadow_policy({
            "route_plan": {"intent": intent, "route": route, "risk_level": "L1"},
            "supervisor_decision": {"next_expected_node": route[0]},
        }, route[0])

        assert policy["control_eligible"] is False
        assert intent_blocker in policy["control_blockers"]
        assert agent_blocker in policy["control_blockers"]
        assert "supervisor_control_blocked" in policy["policy_warnings"]


def test_waiting_approval_and_pending_fields_are_never_control_eligible():
    approval_payload = {"approval_id": "a1"}
    state = {
        "status": "waiting_approval",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
        "completed_nodes": [],
        "approval_payload": approval_payload,
        "approval_required": True,
        "pending_approval_id": "a1",
        "pending_tool_call_id": 7,
        "pending_tool_name": "email.send",
        "supervisor_decision": {"next_expected_node": "tool_agent"},
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == _END_SENTINEL
    policy = state["supervisor_shadow_policy"]
    assert policy["control_eligible"] is False
    assert "waiting_approval" in policy["control_blockers"]
    assert "approval_pending" in policy["control_blockers"]
    assert "pending_tool_or_approval" in policy["control_blockers"]
    assert "risk_level:L3" in policy["control_blockers"]
    assert "graph_interrupt" in policy["control_blockers"]


def test_shadow_policy_does_not_mutate_protected_state_fields():
    state = {
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "execution_plan": {"tasks": []},
        "completed_nodes": [],
        "status": "completed",
        "approval_payload": {"approval_id": "a1"},
        "pending_tool_call_id": None,
        "final_output": "answer",
        "supervisor_decision": {"next_expected_node": None},
    }
    protected_keys = (
        "route_plan",
        "execution_plan",
        "completed_nodes",
        "status",
        "approval_payload",
        "pending_tool_call_id",
        "final_output",
    )
    before = {key: copy.deepcopy(state.get(key)) for key in protected_keys}

    policy = build_supervisor_shadow_policy(state, "final_response")

    assert policy["mode"] == "shadow_only"
    for key, value in before.items():
        assert state.get(key) == value


def test_graph_writes_shadow_policy_without_changing_legacy_next_node():
    state = {
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "completed_nodes": [],
        "supervisor_decision": {"next_expected_node": None},
    }

    next_node = _runtime()._dispatch_next_route_node(state)

    assert next_node == "final_response"
    assert state["supervisor_shadow_policy"]["mode"] == "shadow_only"
    assert state["supervisor_shadow_policy"]["legacy_next_node"] == "final_response"
    assert state["supervisor_shadow_policy"]["recommended_next_node"] == "final_response"
