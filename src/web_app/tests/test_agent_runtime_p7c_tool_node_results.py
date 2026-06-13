import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.dispatch import END_SENTINEL, dispatch_next_route_node
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


@pytest.mark.asyncio
async def test_waiting_approval_records_needs_approval_without_completion(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes, "infer_tool", lambda *args, **kwargs: ("email.send", {"to": "a@example.com"}))
    monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *args, **kwargs: ({"to": "a@example.com"}, []))
    monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *args, **kwargs: {
        "id": "tc1",
        "status": "waiting_approval",
        "approval_id": "ap1",
        "output": {"preview": "send"},
    })

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

    assert result["status"] == "waiting_approval"
    assert result["approval_payload"]["approval_id"] == "ap1"
    assert result["pending_tool_name"] == "email.send"
    assert result["pending_tool_call_id"] == "tc1"
    assert "tool_agent" not in result["completed_nodes"]
    assert latest_agent_result(result, "tool_agent")["status"] == "needs_approval"
    assert _node_result(result)["node"] == "tool_agent"
    assert _node_result(result)["status"] == "needs_approval"
    assert _node_result(result)["delta"]["updates"]["approval_payload"] == result["approval_payload"]
    assert dispatch_next_route_node(result) == END_SENTINEL


@pytest.mark.asyncio
async def test_resume_approved_records_ok_and_clears_pending(monkeypatch):
    _patch_common(monkeypatch)
    state = _base_state()
    state.update({
        "pending_tool_call_id": "tc1",
        "resolved_tool_call_ids": ["tc1"],
        "approval_required": True,
        "approval_payload": {"approval_id": "ap1"},
        "pending_approval_id": "ap1",
        "pending_tool_name": "email.send",
        "pending_tool_args": {"to": "a@example.com"},
        "resume_token": "approval:ap1",
        "tool_call": {"id": "tc1", "status": "completed", "tool_name": "email.send"},
        "tool_result": {"status": "completed"},
    })

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(state)

    assert "tool_agent" in result["completed_nodes"]
    assert result["approval_required"] is False
    assert result["approval_payload"] is None
    assert result["pending_tool_call_id"] is None
    assert latest_agent_result(result, "tool_agent")["status"] == "ok"
    assert _node_result(result)["status"] == "ok"


@pytest.mark.asyncio
async def test_resume_failed_and_rejected_record_failed_or_denied(monkeypatch):
    _patch_common(monkeypatch)

    failed_state = _base_state()
    failed_state.update({
        "pending_tool_call_id": "tc1",
        "_resume_context": "failed:tc1",
        "tool_call": {"id": "tc1", "tool_name": "email.send"},
        "tool_result": {"message": "smtp down"},
    })
    failed = await RuntimeNodes(make_test_session(), {}).tool_agent(failed_state)
    assert "tool_agent" in failed["completed_nodes"]
    assert latest_agent_result(failed, "tool_agent")["status"] == "failed"
    assert _node_result(failed)["status"] == "failed"

    rejected_state = _base_state()
    rejected_state.update({
        "pending_tool_call_id": "tc2",
        "_resume_context": "rejected:tc2",
        "tool_call": {"id": "tc2", "tool_name": "email.send"},
    })
    rejected = await RuntimeNodes(make_test_session(), {}).tool_agent(rejected_state)
    assert "tool_agent" in rejected["completed_nodes"]
    assert rejected["tool_call"]["status"] == "rejected"
    assert latest_agent_result(rejected, "tool_agent")["status"] == "denied"
    assert _node_result(rejected)["status"] == "denied"


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
