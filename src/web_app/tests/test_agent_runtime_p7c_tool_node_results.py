import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.node_groups import agent_nodes, eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state_delta import latest_agent_result
from src.web_app.tests.db_test_utils import make_test_session


def _patch_common(monkeypatch):
    monkeypatch.setattr(agent_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "llm_select_tools", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skip llm")))


def _base_state():
    return {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "tool",
        "user_input": "send a test email",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
        "completed_nodes": [],
        "agent_results": [],
    }


def _node_result(state):
    return state["node_results"][-1]


# ── updated for interrupt-based approval ──────────────────────────


@pytest.mark.asyncio
async def test_interrupt_pause_sets_approval_payload_and_interrupts(monkeypatch):
    """When mcp_service returns waiting_approval, tool_agent calls
    LangGraph interrupt() with the correct payload containing
    approval_id, tool_call_id, tool_name, risk_level."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes, "infer_tool", lambda *args, **kwargs: ("email.send", {"to": "a@example.com"}))
    monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *args, **kwargs: ({"to": "a@example.com"}, []))
    monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *args, **kwargs: {
        "id": "tc1",
        "status": "waiting_approval",
        "approval_id": "ap1",
        "output": {"preview": "send"},
    })

    # Simulate interrupt + approved resume
    interrupt_captured = []
    def _fake_interrupt(payload):
        interrupt_captured.append(payload)
        return {"action": "approved", "tool_result": {"success": True, "sent": True}}

    import langgraph.types as lg_types
    monkeypatch.setattr(lg_types, "interrupt", _fake_interrupt)

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

    # interrupt was called
    assert len(interrupt_captured) == 1
    payload = interrupt_captured[0]
    assert payload["type"] == "approval_required"
    assert payload["approval_pause_mode"] == "interrupt"
    assert payload["tool_name"] == "email.send"
    assert payload["approval_id"] == "ap1"
    assert payload["tool_call_id"] == "tc1"
    assert payload["risk_level"] == "L3"

    # After approved resume: completed + clean
    assert "tool_agent" in result["completed_nodes"]
    assert result["tool_result"]["success"] is True
    assert result["approval_required"] is False
    assert result["pending_tool_call_id"] is None
    assert latest_agent_result(result, "tool_agent")["status"] == "ok"


@pytest.mark.asyncio
async def test_interrupt_resume_approved_clears_pending_and_completes(monkeypatch):
    """When interrupt() returns action=approved, tool_agent accepts
    the tool_result and marks itself completed."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes, "infer_tool", lambda *args, **kwargs: ("email.send", {"to": "a@example.com"}))
    monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *args, **kwargs: ({"to": "a@example.com"}, []))
    monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *args, **kwargs: {
        "id": "tc1",
        "status": "waiting_approval",
        "approval_id": "ap1",
        "output": {},
    })

    import langgraph.types as lg_types_approved
    monkeypatch.setattr(lg_types_approved, "interrupt",
        lambda payload: {"action": "approved", "tool_result": {"success": True, "provider": "mock", "to": "a@b.com"}}
    )

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

    assert "tool_agent" in result["completed_nodes"]
    assert result["approval_required"] is False
    assert result["approval_payload"] is None
    assert result["pending_tool_call_id"] is None
    assert result["pending_tool_name"] is None
    assert result["pending_approval_id"] is None
    assert result["resume_token"] is None
    assert result["tool_result"] == {"success": True, "provider": "mock", "to": "a@b.com"}
    assert result["tool_call"]["status"] == "completed"
    assert latest_agent_result(result, "tool_agent")["status"] == "ok"
    assert _node_result(result)["status"] == "ok"


@pytest.mark.asyncio
async def test_interrupt_resume_rejected_records_denied(monkeypatch):
    """When interrupt() returns action=rejected, tool_agent records
    the rejection without executing the tool."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes, "infer_tool", lambda *args, **kwargs: ("email.send", {"to": "a@example.com"}))
    monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *args, **kwargs: ({"to": "a@example.com"}, []))
    monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *args, **kwargs: {
        "id": "tc2",
        "status": "waiting_approval",
        "approval_id": "ap2",
        "output": {},
    })

    import langgraph.types as lg_types_reject
    monkeypatch.setattr(lg_types_reject, "interrupt",
        lambda payload: {"action": "rejected", "reason": "User rejected the approval"}
    )

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

    assert "tool_agent" in result["completed_nodes"]
    assert result["tool_call"]["status"] == "rejected"
    assert result["tool_result"]["status"] == "rejected"
    assert "User rejected" in result["tool_result"]["message"]
    assert result["approval_required"] is False
    assert result["pending_tool_call_id"] is None
    assert latest_agent_result(result, "tool_agent")["status"] == "denied"
    assert _node_result(result)["status"] == "denied"


# ── unchanged tests (no approval pause path) ──────────────────────


@pytest.mark.asyncio
async def test_missing_fields_records_skipped_and_preserves_final_output(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes, "infer_tool", lambda *args, **kwargs: ("email.send", {"to": "a@example.com"}))
    monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *args, **kwargs: (
        {"to": "a@example.com"},
        [{"field": "subject", "question": "subject?"}],
    ))

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

    assert "tool_agent" in result["completed_nodes"]
    assert result["tool_call"]["status"] == "missing_fields"
    assert result["final_output"]
    assert latest_agent_result(result, "tool_agent")["status"] == "skipped"
    assert _node_result(result)["status"] == "skipped"
    assert _node_result(result)["delta"]["updates"]["final_output"] == result["final_output"]


@pytest.mark.asyncio
async def test_tool_not_found_and_execution_failures_record_failed_or_denied(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes, "infer_tool", lambda *args, **kwargs: (None, {}))
    monkeypatch.setattr(agent_nodes, "_is_obvious_email_intent", lambda *args, **kwargs: False)

    not_found = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())
    assert "tool_agent" in not_found["completed_nodes"]
    assert latest_agent_result(not_found, "tool_agent")["status"] == "failed"
    assert _node_result(not_found)["status"] == "failed"

    monkeypatch.setattr(agent_nodes, "infer_tool", lambda *args, **kwargs: ("email.send", {"to": "a@example.com"}))
    monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *args, **kwargs: ({"to": "a@example.com"}, []))
    monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *args, **kwargs: {
        "status": "blocked",
        "error": "risk denied",
    })

    blocked = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())
    assert "tool_agent" in blocked["completed_nodes"]
    assert latest_agent_result(blocked, "tool_agent")["status"] == "denied"
    assert _node_result(blocked)["status"] == "denied"


@pytest.mark.asyncio
async def test_blocked_route_records_skipped_node_result(monkeypatch):
    _patch_common(monkeypatch)
    state = _base_state()
    state["route"] = "blocked"

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(state)

    assert "tool_agent" in result["completed_nodes"]
    assert _node_result(result)["status"] == "skipped"


@pytest.mark.asyncio
async def test_final_response_still_does_not_put_node_results_in_payload(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route": "chat",
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "final_output": "answer",
        "node_results": [{"node": "previous_node"}],
    }

    result = await RuntimeNodes(make_test_session(), {}).final_response(state)

    assert result["final_payload"]["answer"] == "answer"
    assert "node_results" not in result["final_payload"]
    assert result["node_results"][-1]["node"] == "final_response"
