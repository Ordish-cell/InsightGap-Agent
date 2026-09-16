from src.web_app.agent.runtime.checkpointers import build_checkpointer
from src.web_app.db.repositories.approval_repository import ApprovalRepository
import pytest

pytestmark = pytest.mark.usefixtures("scripted_supervisor")

import json

from sqlalchemy.orm import sessionmaker

from src.web_app.agent.runtime.ledger_stream import stream_ledger_events
from src.web_app.services.agent_service import replay_events, run_agent, run_agent_async
from src.web_app.services.approval_service import update_approval_status
from src.web_app.models.orm import User
from src.web_app.tests.db_test_utils import make_test_session, configure_test_model


def _user(db, email="runtime-stage1@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    configure_test_model(db, user)
    return user


def test_chat_run_emits_agent_events():
    db = make_test_session()
    user = _user(db)

    result = run_agent(db, user.id, {"user_input": "hello"})

    assert result["status"] == "completed"
    assert result["intent"] == "chat"
    events = replay_events(db, user.id, result["run_id"], limit=500)["events"]
    names = [event["event_type"] for event in events]
    assert "run_started" in names
    assert "run_completed" in names
    assert any(event["event_type"] == "node_completed" for event in events)


def test_chat_run_emits_visible_thoughts_without_react_fields():
    db = make_test_session()
    user = _user(db, "visible-thoughts@example.com")

    result = run_agent(db, user.id, {"user_input": "杭州有啥好玩的？"})

    thoughts = result["final_response"]["thinking_summary"]
    assert isinstance(thoughts, list)
    assert isinstance(result["visible_thoughts"], list)
    assert "answer" in result["final_response"]
    assert all("planner" not in item and "context_builder" not in item for item in thoughts)

    events = replay_events(db, user.id, result["run_id"], limit=500)["events"]
    visible_events = [event for event in events if event["event_type"] == "visible_thought"]
    for event in visible_events:
        assert "text" in event["payload"]
        assert not {"action", "observation", "next_action"} & event["payload"].keys()


@pytest.mark.asyncio
async def test_agent_run_stream_emits_visible_thought_and_answer_deltas_before_completion():
    db = make_test_session()
    user = _user(db, "visible-stream@example.com")

    result = await run_agent_async(db, user.id, {"user_input": "hello"})
    session_factory = sessionmaker(bind=db.bind, future=True)
    event_types = []
    first_thought_delta = None
    first_answer_delta = None
    async for chunk in stream_ledger_events(session_factory, user.id, result["run_id"]):
        data = next((line[6:] for line in chunk.splitlines() if line.startswith("data: ")), "")
        if not data:
            continue
        event = json.loads(data)
        event_type = event["event_type"]
        event_types.append(event_type)
        if event_type == "visible_thought_delta" and first_thought_delta is None:
            first_thought_delta = event["payload"]
        if event_type == "answer_delta" and first_answer_delta is None:
            first_answer_delta = event["payload"]
        if event_type == "run_completed":
            break

    assert "answer_started" in event_types
    assert "answer_delta" in event_types
    assert "answer_completed" in event_types
    assert "run_completed" in event_types
    assert event_types.index("answer_started") < event_types.index("answer_delta")
    assert event_types.index("answer_delta") < event_types.index("answer_completed")
    assert event_types.index("answer_completed") < event_types.index("run_completed")
    assert first_answer_delta and len(first_answer_delta["text"]) >= 1


def test_explicit_research_uses_supervisor_action():
    db = make_test_session()
    user = _user(db, "intent-research@example.com")

    result = run_agent(db, user.id, {"user_input": "Research LangGraph", "route": "research"})
    assert result["research"]["status"] == "completed"
    assert result["route_plan"] == []
    events = replay_events(db, user.id, result["run_id"], limit=500)["events"]
    assert any(e["event_type"] == "supervisor_action" and e["payload"]["action"] == "deep_research" for e in events)


def test_l3_email_task_creates_approval_and_events():
    db = make_test_session()
    user = _user(db, "approval-email@example.com")

    result = run_agent(db, user.id, {"user_input": "帮我给 Leo 发邮件说 demo 明天上午发", "route": "tool", "tool_name": "email.send", "tool_input": {"to": "leo@example.test", "subject": "demo", "body": "tomorrow"}})

    assert result["status"] == "waiting_approval"
    assert result["approval_required"] is True
    assert result["risk_level"] == "L3"
    approvals = ApprovalRepository(db).list_by_user(user.id)
    assert approvals
    events = replay_events(db, user.id, result["run_id"], limit=500)["events"]
    assert any(event["event_type"] == "approval_required" for event in events)


def test_l4_delete_task_is_blocked_without_approval():
    db = make_test_session()
    user = _user(db, "approval-delete@example.com")

    # Even an administrator-enabled L4 tool remains blocked by execution policy.
    from src.web_app.services.mcp_service import mcp_service
    from src.web_app.db.repositories.mcp_repository import MCPToolRepository
    mcp_service.ensure_builtin_tools(db)
    tool = MCPToolRepository(db).get_by_name("local_file.delete")
    tool.enabled = True
    db.commit()

    result = run_agent(db, user.id, {"user_input": "Delete a file", "tool_name": "local_file.delete", "tool_input": {"path": "protected.txt"}})

    assert result["status"] == "completed"
    assert result["risk_level"] == "L4"
    assert result["approval_required"] is False
    assert not ApprovalRepository(db).list_by_user(user.id)


def test_approval_approve_and_reject_update_run_events():
    db = make_test_session()
    user = _user(db, "approval-update@example.com")
    result = run_agent(db, user.id, {"user_input": "send email to Leo", "route": "tool", "tool_name": "email.send", "tool_input": {"to": "leo@example.test", "subject": "demo", "body": "tomorrow"}})
    approval = ApprovalRepository(db).list_by_user(user.id)[0]

    approved = update_approval_status(db, user.id, approval.id, "approved", {"decision": "approved"})

    assert approved["status"] == "approved"
    events = replay_events(db, user.id, result["run_id"], limit=500)["events"]
    assert any(event["event_type"] == "approval_approved" for event in events)
    assert any(event["event_type"] == "approval_approved" for event in events)

    result2 = run_agent(db, user.id, {"user_input": "post comment to website", "route": "tool", "tool_name": "email.send", "tool_input": {"to": "leo@example.test", "subject": "demo", "body": "tomorrow"}})
    approval2 = ApprovalRepository(db).list_by_user(user.id)[0]
    rejected = update_approval_status(db, user.id, approval2.id, "rejected", {"reason": "not now"})
    assert rejected["status"] == "rejected"
    events2 = replay_events(db, user.id, result2["run_id"], limit=500)["events"]
    assert any(event["event_type"] == "approval_rejected" for event in events2)


def test_checkpointer_falls_back_when_redis_unavailable():
    checkpointer = build_checkpointer(backend="redis", redis_url="redis://127.0.0.1:0/0", require_durable=False)
    assert checkpointer is not None
