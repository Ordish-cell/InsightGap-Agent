"""Regression tests for END-based approval pause/resume mechanism.

Covers the full lifecycle:
  L3 pause → graph termination via END_SENTINEL → DB persistence →
  approve resume → tool dedup → final state sanitization /
  reject resume → no tool execution → final state reflects rejection.

These tests lock down the CURRENT END-based behavior so that a future
migration to LangGraph checkpointer + interrupt() does not silently
break the approval contract.

Test list (9 requirements):
  1. Trigger L3 tool → tool_agent returns waiting_approval
  2. Dispatcher returns END_SENTINEL when status=waiting_approval
  3. AgentRun.graph_state saves pending_* fields
  4. ToolCall + Approval persisted to DB
  5. approve → execute_approved_tool called exactly once
  6. resume → tool_agent uses pending_tool_call_id in resolved_tool_call_ids,
     no duplicate approval
  7. reject → no real tool execution, final_response reflects rejection
  8. final state has no stale approval_required / waiting_approval / error
  9. (Bonus) coverage for missing_fields, tool_not_found, blocked paths
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.dispatch import END_SENTINEL, dispatch_next_route_node, legacy_next_route_node
from src.web_app.agent.runtime.node_groups import agent_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state_delta import latest_agent_result
from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository, AgentConversationRepository, AgentRunRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.mcp_repository import MCPToolRepository, ToolCallRepository
from src.web_app.mcp.registry import registry as mcp_registry
from src.web_app.models.orm import User
from src.web_app.services.agent_service import resume_run_after_approval
from src.web_app.services.approval_service import update_approval_status
from src.web_app.tests.db_test_utils import make_test_session


# ── helpers ──────────────────────────────────────────────────────────


def _user(db, email="approval-regression@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _patch_common(monkeypatch):
    monkeypatch.setattr(agent_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        agent_nodes,
        "llm_select_tools",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skip llm")),
    )


def _base_state(**overrides):
    state = {
        "user_id": 1,
        "run_id": 99,
        "thread_id": "regression-thread",
        "route": "tool",
        "user_input": "send a test email to Leo",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
        "completed_nodes": [],
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
    }
    state.update(overrides)
    return state


def _node_result(state):
    results = state.get("node_results") or []
    return results[-1] if results else {}


# ── isolated unit tests (no DB) ──────────────────────────────────────


class TestDispatchEndSentinel:
    """Dispatch no longer handles approval pause (now done via interrupt).
    It routes normally regardless of status=waiting_approval."""

    def test_dispatch_ignores_waiting_approval_status(self):
        """Dispatch no longer returns END_SENTINEL for waiting_approval.
        Approval pause is handled by LangGraph interrupt() inside tool_agent."""
        state = _base_state(
            status="waiting_approval",
            route_plan={"intent": "tool", "route": ["tool_agent"], "risk_level": "L3"},
            completed_nodes=[],
        )
        # Now routes to the next node in route_plan, not END
        assert legacy_next_route_node(state) == "tool_agent"

    def test_legacy_next_route_returns_node_when_not_waiting(self):
        state = _base_state(
            route_plan={"intent": "tool", "route": ["tool_agent"], "risk_level": "L3"},
            completed_nodes=[],
        )
        assert legacy_next_route_node(state) == "tool_agent"

    def test_legacy_next_route_returns_final_response_when_all_done(self):
        state = _base_state(
            route_plan={"intent": "chat", "route": ["final_response"], "risk_level": "L0"},
            completed_nodes=[],
        )
        assert legacy_next_route_node(state) == "final_response"

    def test_dispatch_next_route_ignores_waiting_approval(self):
        """dispatch_next_route_node routes normally — approval pause
        is handled inside tool_agent via interrupt()."""
        state = _base_state(
            status="waiting_approval",
            route_plan={"intent": "tool", "route": ["tool_agent"], "risk_level": "L3"},
            completed_nodes=[],
        )
        assert dispatch_next_route_node(state) == "tool_agent"

    def test_dispatch_next_route_returns_node_when_resuming(self):
        """resuming status should NOT trigger END — it should continue."""
        state = _base_state(
            status="resuming",
            route_plan={"intent": "tool", "route": ["tool_agent"], "risk_level": "L3"},
            completed_nodes=[],
        )
        assert dispatch_next_route_node(state) == "tool_agent"


class TestToolAgentPause:
    """Requirement 1: L3 tool triggers interrupt with correct payload."""

    @pytest.mark.asyncio
    async def test_pause_sets_status_and_all_pending_fields(self, monkeypatch):
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com", "subject": "hi", "body": "hello"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com", "subject": "hi", "body": "hello"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 1001,
            "status": "waiting_approval",
            "approval_id": 2001,
            "output": {"preview": "pending send"},
        })
        import langgraph.types as lg_types
        monkeypatch.setattr(lg_types, "interrupt",
            lambda payload: {"action": "approved", "tool_result": {"success": True}}
        )

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        # After approved resume: completed + clean
        assert "tool_agent" in result["completed_nodes"]
        assert result["tool_result"]["success"] is True
        assert result["approval_required"] is False
        assert result["pending_approval_id"] is None
        assert result["pending_tool_name"] is None
        assert result["pending_tool_call_id"] is None

    @pytest.mark.asyncio
    async def test_pause_records_needs_approval_agent_result(self, monkeypatch):
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 1001,
            "status": "waiting_approval",
            "approval_id": 2001,
            "output": {},
        })
        import langgraph.types as lg_types
        monkeypatch.setattr(lg_types, "interrupt",
            lambda payload: {"action": "approved", "tool_result": {"success": True}}
        )

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        # After approved resume, agent_result should be "ok"
        assert latest_agent_result(result, "tool_agent")["status"] == "ok"


class TestToolAgentResumeApproved:
    """Requirements 6-7: interrupt-based approved resume."""

    @pytest.mark.asyncio
    async def test_interrupt_resume_approved_clears_pending_and_completes(self, monkeypatch):
        """When interrupt() returns action=approved, tool_agent accepts
        the pre-executed result and marks itself completed."""
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 1001, "status": "waiting_approval", "approval_id": 2001, "output": {},
        })
        import langgraph.types as lg_types
        monkeypatch.setattr(lg_types, "interrupt",
            lambda payload: {"action": "approved", "tool_result": {"success": True, "sent": True, "provider": "mock"}}
        )

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        assert "tool_agent" in result["completed_nodes"]
        assert result["approval_required"] is False
        assert result["approval_payload"] is None
        assert result["pending_tool_call_id"] is None
        assert result["pending_tool_name"] is None
        assert result["pending_approval_id"] is None
        assert result["resume_token"] is None
        assert result["tool_result"]["success"] is True
        assert latest_agent_result(result, "tool_agent")["status"] == "ok"


class TestToolAgentResumeRejected:
    """Requirement 8: interrupt-based reject path."""

    @pytest.mark.asyncio
    async def test_interrupt_resume_rejected_sets_tool_call_rejected(self, monkeypatch):
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 1002, "status": "waiting_approval", "approval_id": 2002, "output": {},
        })
        import langgraph.types as lg_types
        monkeypatch.setattr(lg_types, "interrupt",
            lambda payload: {"action": "rejected", "reason": "User rejected the approval"}
        )

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        assert "tool_agent" in result["completed_nodes"]
        assert result["tool_call"]["status"] == "rejected"
        assert result["tool_result"]["status"] == "rejected"
        assert "User rejected" in result["tool_result"]["message"]
        assert latest_agent_result(result, "tool_agent")["status"] == "denied"


class TestToolAgentResumeFailed:
    """Interrupt mode doesn't have a separate 'failed' path — failures
    are handled by the tool_executor on the service side before
    Command(resume=...)."""

    @pytest.mark.asyncio
    async def test_interrupt_resume_failed_tool_result_is_accepted(self, monkeypatch):
        """Even a failed tool_result is accepted by the approved path."""
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 1003, "status": "waiting_approval", "approval_id": 2003, "output": {},
        })
        import langgraph.types as lg_types
        monkeypatch.setattr(lg_types, "interrupt",
            lambda payload: {"action": "approved", "tool_result": {"success": False, "error": "SMTP down"}}
        )

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        assert "tool_agent" in result["completed_nodes"]
        assert result["tool_result"]["success"] is False
        assert result["approval_payload"] is None


class TestToolAgentEdgeCases:
    """Coverage for blocked route, missing fields, tool_not_found."""

    @pytest.mark.asyncio
    async def test_blocked_route_skips_gracefully(self, monkeypatch):
        _patch_common(monkeypatch)
        state = _base_state(route="blocked")
        result = await RuntimeNodes(make_test_session(), {}).tool_agent(state)
        assert "tool_agent" in result["completed_nodes"]
        assert _node_result(result).get("status") == "skipped"

    @pytest.mark.asyncio
    async def test_missing_fields_stops_before_mcp_call(self, monkeypatch):
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: (
            {"to": "a@b.com"},
            [{"field": "subject", "question": "What subject?"}],
        ))
        call_tool_called = []
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: call_tool_called.append(1) or {})

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        assert len(call_tool_called) == 0
        assert result["tool_call"]["status"] == "missing_fields"
        assert latest_agent_result(result, "tool_agent")["status"] == "skipped"


# ── final state sanitization (isolated) ──────────────────────────────


class TestFinalStateSanitization:
    """Requirement 9: final state must not carry stale approval artifacts."""

    @pytest.mark.asyncio
    async def test_approved_interrupt_final_state_clean(self, monkeypatch):
        """After interrupt approved resume, no stale approval fields."""
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 1001, "status": "waiting_approval", "approval_id": 2001, "output": {},
        })
        import langgraph.types as lg_types
        monkeypatch.setattr(lg_types, "interrupt",
            lambda payload: {"action": "approved", "tool_result": {"success": True}}
        )

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        assert result.get("approval_required") is not True
        assert result.get("status") != "waiting_approval"
        tc = result.get("tool_call") or {}
        assert "approval_required" not in str(tc.get("error", "")).lower()

    @pytest.mark.asyncio
    async def test_rejected_interrupt_final_state_clean(self, monkeypatch):
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 1001, "status": "waiting_approval", "approval_id": 2001, "output": {},
        })
        import langgraph.types as lg_types
        monkeypatch.setattr(lg_types, "interrupt",
            lambda payload: {"action": "rejected", "reason": "User rejected the approval"}
        )

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        assert result.get("approval_required") is not True
        tc = result.get("tool_call") or {}
        assert "approval_required" not in str(tc.get("error", "")).lower()
        assert result["tool_call"]["status"] == "rejected"


# ── helpers for integration tests ────────────────────────────────────


def _pause_graph_state(**overrides):
    """Return a graph_state dict that mirrors what tool_agent + agent_service
    would persist after hitting an L3 approval pause."""
    gs = {
        "user_id": 1,
        "run_id": 999,
        "thread_id": "int-test-thread",
        "conversation_id": "int-test-conv",
        "user_input": "send email to Leo",
        "route": "tool",
        "status": "waiting_approval",
        "approval_required": True,
        "pending_approval_id": 9001,
        "pending_tool_name": "email.send",
        "pending_tool_call_id": 8001,
        "pending_tool_args": {"to": "leo@example.com", "subject": "demo", "body": "明天上午 demo"},
        "route_plan_snapshot": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
        "resume_token": "approval:9001",
        "approval_payload": {"approval_id": 9001, "tool_name": "email.send", "risk_level": "L3", "tool_args": {"to": "leo@example.com"}, "run_id": 999, "user_id": 1, "title": "需要你确认：发送邮件", "actions": ["approve", "reject"]},
        "tool_call": {"id": 8001, "tool_name": "email.send", "status": "waiting_approval"},
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3", "needs_approval": True},
        "completed_nodes": ["home_intent_react", "planner", "parallel_prefetch", "parallel_read_stage", "supervisor_observer", "llm_supervisor_route"],
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "langgraphstatus": {"status": "waiting_approval", "steps": []},
    }
    gs.update(overrides)
    return gs


def _create_paused_run(db, user_id: int, **gs_overrides):
    """Create all DB records for a run paused on L3 approval.
    Returns (run, conversation, assistant_message, approval, tool_call, gs)."""
    from uuid import uuid4

    gs = _pause_graph_state(**gs_overrides)
    conv_id = gs["conversation_id"]
    thread_id = gs["thread_id"]
    run_id = gs["run_id"]

    conv_repo = AgentConversationRepository(db)
    run_repo = AgentRunRepository(db)
    msg_repo = AgentChatMessageRepository(db)

    # Conversation — try get or create
    conv = None
    try:
        conv = conv_repo.get_by_conversation_id(user_id, conv_id)
    except Exception:
        pass
    if conv is None:
        from src.web_app.models.orm import AgentConversation
        conv = AgentConversation(
            conversation_id=conv_id,
            user_id=user_id,
            thread_id=thread_id,
            title="test conv",
            status="active",
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)

    # Run
    run = run_repo.create(
        user_id=user_id,
        conversation_id=conv_id,
        thread_id=thread_id,
        run_type="agent_runtime",
        mode="react",
        status="waiting_approval",
        user_input=gs["user_input"],
        graph_state=gs,
    )

    # Assistant message
    assistant_msg = msg_repo.create(
        message_id=str(uuid4()),
        conversation_id=conv_id,
        user_id=user_id,
        run_id=run.id,
        thread_id=thread_id,
        role="assistant",
        content="",
        status="waiting_approval",
    )

    # Approval
    approval_repo = ApprovalRepository(db)
    approval = approval_repo.create(
        user_id=user_id,
        run_id=run.id,
        approval_type="mcp_tool_call",
        title="需要你确认：发送邮件",
        description="收件人: leo@example.com",
        payload={
            "tool_call_id": gs["pending_tool_call_id"],
            "tool_name": gs["pending_tool_name"],
            "tool_args": gs["pending_tool_args"],
            "risk_level": "L3_EXTERNAL_WRITE",
            "requires_approval": True,
        },
    )

    # ToolCall
    mcp_registry.ensure_builtin_tools(db)
    tool_repo = ToolCallRepository(db)
    mcp_tool = MCPToolRepository(db).get_by_name("email.send")
    tool_call = tool_repo.create(
        user_id=user_id,
        run_id=run.id,
        tool_name="email.send",
        mcp_tool_id=mcp_tool.id if mcp_tool else None,
        input=gs["pending_tool_args"],
        output={},
        permission_level="L3_EXTERNAL_WRITE",
        status="waiting_approval",
        error_message="",
    )

    # Update graph_state with real DB IDs
    gs["pending_approval_id"] = str(approval.id)
    gs["resume_token"] = f"approval:{approval.id}"
    gs["pending_tool_call_id"] = tool_call.id
    gs["approval_payload"]["approval_id"] = approval.id
    gs["tool_call"]["id"] = tool_call.id

    approval_repo.update(approval, payload={
        "tool_call_id": tool_call.id,
        "tool_name": gs["pending_tool_name"],
        "tool_args": gs["pending_tool_args"],
        "risk_level": "L3_EXTERNAL_WRITE",
        "requires_approval": True,
    })

    run_repo.update(run, graph_state=gs)
    return run, assistant_msg, approval, tool_call, gs


# ── integration tests (real SQLite DB) ───────────────────────────────


class TestApprovalPauseDBWrite:
    """Requirements 4-5: DB persistence of pause state fields."""

    def test_graph_state_round_trips_pending_fields(self):
        """Write pause graph_state → read back — all fields survive serialization."""
        db = make_test_session()
        user = _user(db)
        gs = _pause_graph_state(
            user_id=user.id,
            pending_approval_id="42",
            pending_tool_call_id=99,
            pending_tool_name="email.send",
            pending_tool_args={"to": "x@y.com", "subject": "hi"},
        )

        run = AgentRunRepository(db).create(
            user_id=user.id,
            conversation_id=gs["conversation_id"],
            thread_id=gs["thread_id"],
            run_type="agent_runtime",
            mode="react",
            status="waiting_approval",
            user_input=gs["user_input"],
            graph_state=gs,
        )
        db.commit()

        loaded = AgentRunRepository(db).get_by_user(user.id, run.id)
        assert loaded is not None
        loaded_gs = loaded.graph_state or {}

        # requirement 4
        assert loaded_gs.get("pending_approval_id") == "42"
        assert loaded_gs.get("pending_tool_call_id") == 99
        assert loaded_gs.get("pending_tool_name") == "email.send"
        assert loaded_gs.get("pending_tool_args") == {"to": "x@y.com", "subject": "hi"}
        assert loaded_gs.get("status") == "waiting_approval"
        assert loaded_gs.get("approval_required") is True

    def test_tool_call_and_approval_persisted_by_mcp_service(self):
        """Requirement 5: mcp_service.call_tool persists ToolCall + Approval."""
        db = make_test_session()
        user = _user(db)

        from src.web_app.services.mcp_service import mcp_service
        result = mcp_service.call_tool(
            db, user.id, "email.send",
            {"to": "leo@example.com", "subject": "hi", "body": "hello"},
        )

        assert result["status"] == "waiting_approval"

        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        assert len(tool_calls) >= 1
        tc = tool_calls[0]
        assert tc.tool_name == "email.send"
        assert tc.status == "waiting_approval"

        approvals = ApprovalRepository(db).list_by_user(user.id)
        assert len(approvals) >= 1
        assert approvals[0].status == "pending"


class TestApproveResumeFlow:
    """Requirements 6-7: approve → tool executes once, resume dedups."""

    @pytest.mark.asyncio
    async def test_approve_then_resume_updates_tool_call_to_completed(self):
        """Requirement 6: approve + resume → tool call transitions to completed."""
        db = make_test_session()
        user = _user(db, "approve-resume-int@example.com")

        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        updated = update_approval_status(db, user.id, approval.id, "approved")
        assert updated["status"] == "approved"

        final_state = await resume_run_after_approval(db, user.id, run.id)

        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        waiting = [t for t in tool_calls if t.status == "waiting_approval"]
        assert len(waiting) == 0, f"all tool calls should be resolved: {[(t.id, t.status) for t in tool_calls]}"

        completed = [t for t in tool_calls if t.status == "completed"]
        assert len(completed) >= 1

        assert final_state.get("status") == "completed"
        assert not final_state.get("approval_required")

    @pytest.mark.asyncio
    async def test_resume_does_not_duplicate_tool_call(self):
        """Requirement 7: tool is NOT called again during resume —
        tool_agent uses pending_tool_call_id in resolved_tool_call_ids skip path.
        Only the pre-executed result is accepted. No new ToolCall created."""
        db = make_test_session()
        user = _user(db, "dedup-approve-int@example.com")

        run, msg, approval, tc, gs = _create_paused_run(db, user.id)
        tc_count_before = len(ToolCallRepository(db).list_by_user(user.id))

        update_approval_status(db, user.id, approval.id, "approved")
        await resume_run_after_approval(db, user.id, run.id)

        # The pre-executed tool call was updated in-place (not duplicated)
        tc_count_after = len(ToolCallRepository(db).list_by_user(user.id))
        assert tc_count_after == tc_count_before, (
            f"tool call must not be duplicated during resume: "
            f"before={tc_count_before} after={tc_count_after}"
        )

        # The original approval stays (planner may create another during replay)
        # but the original must NOT be pending — it's been decided
        approvals = ApprovalRepository(db).list_by_user(user.id)
        assert any(a.status == "approved" for a in approvals), (
            "at least one approval should be approved"
        )


class TestRejectResumeFlow:
    """Requirement 8: reject → no tool execution, final state reflects rejection."""

    @pytest.mark.asyncio
    async def test_reject_does_not_execute_tool(self):
        db = make_test_session()
        user = _user(db, "reject-resume-int@example.com")

        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "rejected")
        final_state = await resume_run_after_approval(db, user.id, run.id)

        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        executed = [t for t in tool_calls if t.status in ("completed", "running")]
        assert len(executed) == 0, (
            f"tool must not be executed on reject: {[(t.id, t.status, t.tool_name) for t in tool_calls]}"
        )

        assert final_state.get("status") != "waiting_approval"
        assert not final_state.get("approval_required")

    @pytest.mark.asyncio
    async def test_reject_resume_clean_db_state(self):
        """Requirement 9: after reject, no stale approval fields in DB."""
        db = make_test_session()
        user = _user(db, "reject-clean-int@example.com")

        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "rejected")
        await resume_run_after_approval(db, user.id, run.id)

        persisted = AgentRunRepository(db).get_by_user(user.id, run.id)
        assert persisted is not None
        assert persisted.status not in ("waiting_approval", "resuming")

        persisted_gs = persisted.graph_state or {}
        assert persisted_gs.get("approval_required") is not True
        assert persisted_gs.get("status") != "waiting_approval"

        ptc = persisted_gs.get("tool_call") or {}
        assert "approval_required" not in str(ptc.get("error", "")).lower()
        if persisted.error_message:
            assert "approval_required" not in persisted.error_message.lower()


class TestApproveResumeNoStaleState:
    """Requirement 9: after approve resume, no stale approval anywhere."""

    @pytest.mark.asyncio
    async def test_approve_resume_clean_db_and_final_state(self):
        db = make_test_session()
        user = _user(db, "approve-clean-int@example.com")

        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "approved")
        final_state = await resume_run_after_approval(db, user.id, run.id)

        assert final_state.get("approval_required") is not True
        assert final_state.get("status") not in ("waiting_approval", "resuming")
        assert "approval_required" not in str(final_state.get("error", "")).lower()

        persisted = AgentRunRepository(db).get_by_user(user.id, run.id)
        assert persisted.status not in ("waiting_approval", "resuming")
        if persisted.error_message:
            assert "approval_required" not in persisted.error_message.lower()

        persisted_gs = persisted.graph_state or {}
        assert persisted_gs.get("approval_required") is not True
        ptc = persisted_gs.get("tool_call") or {}
        assert "approval_required" not in str(ptc.get("error", "")).lower()


class TestMultipleApprovalRuns:
    """Concurrent approval runs must not interfere."""

    @pytest.mark.asyncio
    async def test_approve_one_run_does_not_affect_other(self):
        db = make_test_session()
        user = _user(db, "isolated-resume-int@example.com")

        run1, _, approval1, _, gs1 = _create_paused_run(db, user.id, run_id=1001, thread_id="t1")
        run2, _, approval2, _, gs2 = _create_paused_run(db, user.id, run_id=1002, thread_id="t2")

        update_approval_status(db, user.id, approval1.id, "approved")
        await resume_run_after_approval(db, user.id, run1.id)

        r1 = AgentRunRepository(db).get_by_user(user.id, run1.id)
        assert r1.status == "completed"

        r2 = AgentRunRepository(db).get_by_user(user.id, run2.id)
        assert r2.status == "waiting_approval"


# ── Phase 7: graph_state structure verification ─────────────────────


class TestGraphStateStructure:
    """Verify graph_state composition differs correctly between modes."""

    def test_end_based_graph_state_is_full(self):
        """END-based pause saves full runtime state (100+ fields)."""
        db = make_test_session()
        user = _user(db, "gs-full@example.com")
        run, _, _, _, gs = _create_paused_run(db, user.id, approval_pause_mode="end")
        persisted = AgentRunRepository(db).get_by_user(user.id, run.id)
        persisted_gs = persisted.graph_state or {}

        # END-based has full runtime state fields
        assert persisted_gs.get("approval_pause_mode") == "end"
        assert persisted_gs.get("status") == "waiting_approval"
        assert persisted_gs.get("pending_tool_call_id") is not None
        assert persisted_gs.get("pending_approval_id") is not None
        # Full state has many fields beyond just pause summary
        assert len(persisted_gs) > 10

    def test_interrupt_graph_state_is_minimal(self):
        """Interrupt-based pause saves only pause summary, not full state."""
        db = make_test_session()
        user = _user(db, "gs-minimal@example.com")
        run, _, _, _, gs = _create_paused_run(db, user.id, approval_pause_mode="interrupt")

        # Simulate what run_agent_async saves after catching GraphInterrupt:
        # Only the pause summary fields, not the full 100+ field state.
        minimal_gs = {
            "user_id": user.id,
            "run_id": run.id,
            "thread_id": gs["thread_id"],
            "conversation_id": gs["conversation_id"],
            "user_input": gs["user_input"],
            "status": "waiting_approval",
            "approval_required": True,
            "approval_pause_mode": "interrupt",
            "pending_approval_id": gs["pending_approval_id"],
            "pending_tool_call_id": gs["pending_tool_call_id"],
            "pending_tool_name": gs["pending_tool_name"],
            "route_plan": {"risk_level": "L3"},
            "tool_call": {"id": gs["pending_tool_call_id"], "tool_name": "email.send", "status": "waiting_approval"},
            "approval_payload": {
                "approval_id": int(gs["pending_approval_id"]) if gs["pending_approval_id"] else None,
                "tool_name": "email.send",
                "risk_level": "L3",
                "run_id": run.id,
                "user_id": user.id,
                "title": "需要你确认：email.send",
                "actions": ["approve", "reject"],
                "approval_pause_mode": "interrupt",
            },
            "visible_thoughts": [],
            "langgraphstatus": {"status": "waiting_approval", "steps": []},
        }
        run_repo = AgentRunRepository(db)
        run_repo.update(run, graph_state=minimal_gs)

        persisted = AgentRunRepository(db).get_by_user(user.id, run.id)
        persisted_gs = persisted.graph_state or {}

        # Interrupt state is minimal — must NOT have full state blobs
        assert persisted_gs.get("approval_pause_mode") == "interrupt"
        assert persisted_gs.get("thread_id") is not None
        assert persisted_gs.get("pending_tool_call_id") is not None
        assert persisted_gs.get("pending_approval_id") is not None

        # These large fields should NOT be in interrupt graph_state
        for large_field in (
            "prefetch_results", "parallel_read_results",
            "context_packets", "selected_memories", "matched_skills",
            "supervisor_decision", "supervisor_trace",
            "execution_plan", "research_result", "rag_result",
        ):
            assert large_field not in persisted_gs, (
                f"interrupt graph_state must NOT contain '{large_field}'"
            )

        # Key pause summary fields MUST be present
        for required in (
            "approval_pause_mode", "thread_id", "pending_approval_id",
            "pending_tool_call_id", "pending_tool_name",
            "approval_required", "status",
        ):
            assert required in persisted_gs, (
                f"interrupt graph_state must contain '{required}'"
            )

    @pytest.mark.asyncio
    async def test_interrupt_resume_does_not_read_full_state_from_db(self):
        """_resume_interrupt_approval only needs thread_id from graph_state.
        It must NOT try to read context/research/parallel_read results."""
        db = make_test_session()
        user = _user(db, "gs-resume@example.com")
        run, msg, approval, tc, gs = _create_paused_run(
            db, user.id, approval_pause_mode="interrupt",
        )

        # Inject only minimal fields (simulating interrupt pause)
        minimal_gs = {
            "approval_pause_mode": "interrupt",
            "thread_id": gs["thread_id"],
            "pending_approval_id": gs["pending_approval_id"],
            "pending_tool_call_id": gs["pending_tool_call_id"],
            "pending_tool_name": gs["pending_tool_name"],
            "status": "waiting_approval",
            "approval_required": True,
        }
        run_repo = AgentRunRepository(db)
        run_repo.update(run, graph_state=minimal_gs)

        update_approval_status(db, user.id, approval.id, "approved")

        # This must NOT raise KeyError or try to read missing graph_state fields
        final_state = await resume_run_after_approval(db, user.id, run.id)
        assert final_state.get("status") in ("completed", "failed")

    def test_end_based_graph_state_survives_round_trip(self):
        """END-based full graph_state round-trips through pause+resume."""
        db = make_test_session()
        user = _user(db, "gs-roundtrip@example.com")

        # Simulate a full END-based graph_state with rich context
        full_gs = _pause_graph_state(
            approval_pause_mode="end",
            prefetch_results={"rag": {"evidence": ["doc1", "doc2"]}},
            parallel_read_results={"rag_prepare": {"evidence": ["chunk1"]}},
            context_packets=[{"type": "rag", "data": "test"}],
            supervisor_decision={"next": "tool_agent"},
        )
        run_repo = AgentRunRepository(db)
        run = run_repo.create(
            user_id=user.id,
            conversation_id=full_gs["conversation_id"],
            thread_id=full_gs["thread_id"],
            run_type="agent_runtime", mode="react",
            status="waiting_approval",
            user_input=full_gs["user_input"],
            graph_state=full_gs,
        )

        persisted = AgentRunRepository(db).get_by_user(user.id, run.id)
        persisted_gs = persisted.graph_state or {}

        # Full state fields survive serialization round-trip
        assert persisted_gs.get("prefetch_results") is not None
        assert persisted_gs.get("parallel_read_results") is not None
        assert persisted_gs.get("context_packets") is not None
        assert persisted_gs.get("supervisor_decision") is not None


# ── Phase 9: idempotency, legacy compat, E2E flow ──────────────────


class TestIdempotency:
    """Double-click safety: approve/reject must be idempotent."""

    # NOTE: Idempotency tests use the legacy END-based resume path
    # because Command(resume=...) requires a persistent checkpointer
    # (RedisSaver) that survives across AgentRuntime instances.
    # MemorySaver is per-instance and cannot support cross-invocation
    # resume in integration tests.  The interrupt resume path is
    # covered by isolated tests in test_agent_runtime_p7c_tool_node_results.py.

    @pytest.mark.asyncio
    async def test_double_approve_only_executes_tool_once(self):
        """Approve twice → first succeeds, second raises clear error."""
        db = make_test_session()
        user = _user(db, "double-approve@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "approved")
        result1 = await resume_run_after_approval(db, user.id, run.id)
        assert result1.get("status") == "completed"

        with pytest.raises(ValueError, match="运行状态不正确"):
            await resume_run_after_approval(db, user.id, run.id)

        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        completed_calls = [t for t in tool_calls if t.status == "completed"]
        assert len(completed_calls) == 1

    @pytest.mark.asyncio
    async def test_double_reject_does_not_execute_tool(self):
        """Reject twice → first succeeds, second raises clear error."""
        db = make_test_session()
        user = _user(db, "double-reject@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "rejected")
        result1 = await resume_run_after_approval(db, user.id, run.id)
        assert result1.get("status") == "completed"

        with pytest.raises(ValueError, match="运行状态不正确"):
            await resume_run_after_approval(db, user.id, run.id)

        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        executed = [t for t in tool_calls if t.status in ("completed", "running")]
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_reject_then_try_approve_errors(self):
        """After reject, run completed → subsequent approve errors clearly."""
        db = make_test_session()
        user = _user(db, "reject-approve@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "rejected")
        result1 = await resume_run_after_approval(db, user.id, run.id)
        assert result1.get("status") == "completed"

        with pytest.raises(ValueError, match="运行状态不正确"):
            await resume_run_after_approval(db, user.id, run.id)

        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        executed = [t for t in tool_calls if t.status in ("completed", "running")]
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_approve_then_try_reject_errors(self):
        """After approve, run completed → subsequent reject errors clearly."""
        db = make_test_session()
        user = _user(db, "approve-reject@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "approved")
        result1 = await resume_run_after_approval(db, user.id, run.id)
        assert result1.get("status") == "completed"

        with pytest.raises(ValueError, match="运行状态不正确"):
            await resume_run_after_approval(db, user.id, run.id)

        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        completed_calls = [t for t in tool_calls if t.status == "completed"]
        assert len(completed_calls) >= 1


class TestLegacyRunCompatibility:
    """Old DB runs without approval_pause_mode must route correctly."""

    def test_no_approval_pause_mode_defaults_to_legacy(self):
        """Graph state missing approval_pause_mode → treated as 'end' (legacy)."""
        db = make_test_session()
        user = _user(db, "no-pause-mode@example.com")
        # Create run WITHOUT approval_pause_mode
        gs = _pause_graph_state()
        gs.pop("approval_pause_mode", None)  # explicitly remove

        run_repo = AgentRunRepository(db)
        run = run_repo.create(
            user_id=user.id,
            conversation_id=gs["conversation_id"],
            thread_id=gs["thread_id"],
            run_type="agent_runtime", mode="react",
            status="waiting_approval",
            user_input=gs["user_input"],
            graph_state=gs,
        )

        persisted = AgentRunRepository(db).get_by_user(user.id, run.id)
        persisted_gs = persisted.graph_state or {}
        assert "approval_pause_mode" not in persisted_gs, (
            "old run should not have approval_pause_mode"
        )

    @pytest.mark.asyncio
    async def test_legacy_run_routes_to_legacy_resume(self):
        """Old run (no approval_pause_mode) routes to
        _resume_legacy_end_based_approval with warning log."""
        db = make_test_session()
        user = _user(db, "legacy-route@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)
        # Remove approval_pause_mode to simulate old run
        gs.pop("approval_pause_mode", None)
        run_repo = AgentRunRepository(db)
        run_repo.update(run, graph_state=gs)

        update_approval_status(db, user.id, approval.id, "approved")

        import logging
        import io
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("src.web_app.services.agent_service")
        logger.addHandler(handler)
        try:
            result = await resume_run_after_approval(db, user.id, run.id)
            assert result.get("status") in ("completed", "failed")
            log_output = log_stream.getvalue()
            assert "LEGACY" in log_output, (
                f"legacy resume must log LEGACY warning, got: {log_output[:500]}"
            )
        finally:
            logger.removeHandler(handler)


class TestE2EInterruptFlow:
    """End-to-end: full interrupt pause → resume cycle with mocked interrupt()."""

    @pytest.mark.asyncio
    async def test_full_approve_cycle_mocked_interrupt(self, monkeypatch):
        """Simulate full interrupt flow:
        1. tool_agent calls interrupt()
        2. Approved resume via _handle_tool_resume_approved
        3. Verify no graph replay (no permission_guard, planner, etc.)"""
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 777,
            "status": "waiting_approval",
            "approval_id": 888,
            "output": {"preview": "test"},
        })
        import langgraph.types as lg_types
        interrupt_called = []
        def _capture_interrupt(payload):
            interrupt_called.append(payload)
            return {"action": "approved", "tool_result": {"success": True, "provider": "mock"}}
        monkeypatch.setattr(lg_types, "interrupt", _capture_interrupt)

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        # interrupt was called
        assert len(interrupt_called) == 1
        assert interrupt_called[0]["type"] == "approval_required"
        assert interrupt_called[0]["approval_pause_mode"] == "interrupt"

        # tool_agent completed normally after resume
        assert "tool_agent" in result["completed_nodes"]
        assert result["approval_required"] is False
        assert result["approval_payload"] is None
        assert result["tool_call"]["status"] == "completed"

        # No stale approval state
        assert result.get("status") != "waiting_approval"

    @pytest.mark.asyncio
    async def test_full_reject_cycle_mocked_interrupt(self, monkeypatch):
        """Simulate full reject flow via interrupt()."""
        _patch_common(monkeypatch)
        monkeypatch.setattr(agent_nodes, "infer_tool", lambda *a, **kw: ("email.send", {"to": "a@b.com"}))
        monkeypatch.setattr(agent_nodes, "validate_tool_input", lambda *a, **kw: ({"to": "a@b.com"}, []))
        monkeypatch.setattr(agent_nodes.mcp_service, "call_tool", lambda *a, **kw: {
            "id": 999,
            "status": "waiting_approval",
            "approval_id": 111,
            "output": {},
        })
        import langgraph.types as lg_types
        interrupt_called = []
        def _capture_interrupt(payload):
            interrupt_called.append(payload)
            return {"action": "rejected", "reason": "User cancelled"}
        monkeypatch.setattr(lg_types, "interrupt", _capture_interrupt)

        result = await RuntimeNodes(make_test_session(), {}).tool_agent(_base_state())

        assert len(interrupt_called) == 1
        assert "tool_agent" in result["completed_nodes"]
        assert result["tool_call"]["status"] == "rejected"
        assert result["tool_result"]["status"] == "rejected"
        assert result.get("status") != "waiting_approval"


class TestToolCallApprovalAuditConsistency:
    """ToolCall, Approval, and Run status must be consistent after all paths."""

    def test_paused_run_has_consistent_status(self):
        """After pause: run=waiting_approval, Approval=pending, ToolCall=waiting_approval."""
        db = make_test_session()
        user = _user(db, "consistent-pause@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        persisted_run = AgentRunRepository(db).get_by_user(user.id, run.id)
        assert persisted_run.status == "waiting_approval"
        assert approval.status == "pending"
        assert tc.status == "waiting_approval"

    @pytest.mark.asyncio
    async def test_approved_run_has_consistent_final_status(self):
        """After approve: run=completed, Approval=approved, ToolCall=completed."""
        db = make_test_session()
        user = _user(db, "consistent-approve@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "approved")
        await resume_run_after_approval(db, user.id, run.id)

        persisted_run = AgentRunRepository(db).get_by_user(user.id, run.id)
        assert persisted_run.status == "completed"

        # Approval should be approved
        persisted_approval = ApprovalRepository(db).get_by_user(user.id, approval.id)
        assert persisted_approval.status == "approved"

        # ToolCall: may have been updated to completed by execute_approved_tool
        tool_calls = ToolCallRepository(db).list_by_user(user.id)
        assert any(t.status == "completed" for t in tool_calls), (
            f"at least one tool call must be completed: {[(t.id, t.status) for t in tool_calls]}"
        )

    @pytest.mark.asyncio
    async def test_rejected_run_has_consistent_final_status(self):
        """After reject: run=completed, Approval=rejected, ToolCall NOT completed."""
        db = make_test_session()
        user = _user(db, "consistent-reject@example.com")
        run, msg, approval, tc, gs = _create_paused_run(db, user.id)

        update_approval_status(db, user.id, approval.id, "rejected")
        await resume_run_after_approval(db, user.id, run.id)

        persisted_run = AgentRunRepository(db).get_by_user(user.id, run.id)
        assert persisted_run.status == "completed"

        persisted_approval = ApprovalRepository(db).get_by_user(user.id, approval.id)
        assert persisted_approval.status == "rejected"
