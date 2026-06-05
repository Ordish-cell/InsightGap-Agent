from src.web_app.agent.runtime.checkpointers import build_checkpointer
from src.web_app.db.repositories.approval_repository import ApprovalRepository
import pytest

from src.web_app.services.agent_service import list_events, run_agent, stream_agent_run
from src.web_app.services.approval_service import update_approval_status
from src.web_app.models.orm import User
from src.web_app.tests.db_test_utils import make_test_session


def _user(db, email="runtime-stage1@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_chat_run_emits_agent_events():
    db = make_test_session()
    user = _user(db)

    result = run_agent(db, user.id, {"user_input": "hello"})

    assert result["status"] == "completed"
    assert result["intent"] == "chat"
    events = list_events(db, user.id, result["run_id"])
    names = [event["event"] for event in events]
    assert "run_started" in names
    assert "run_completed" in names
    assert any(event["data"]["event_type"] == "node_completed" for event in events)


def test_chat_run_emits_visible_thoughts_without_react_fields():
    db = make_test_session()
    user = _user(db, "visible-thoughts@example.com")

    result = run_agent(db, user.id, {"user_input": "杭州有啥好玩的？"})

    thoughts = result["final_response"]["thinking_summary"]
    assert thoughts
    assert result["visible_thoughts"]
    assert "answer" in result["final_response"]
    assert all("planner" not in item and "context_builder" not in item for item in thoughts)

    events = list_events(db, user.id, result["run_id"])
    visible_events = [event for event in events if event["event"] == "visible_thought"]
    assert visible_events
    assert visible_events[0]["data"]["payload"]["status"] == "completed"
    assert "text" in visible_events[0]["data"]["payload"]
    assert "action" not in visible_events[0]["data"]["payload"]
    assert "observation" not in visible_events[0]["data"]["payload"]
    assert "next_action" not in visible_events[0]["data"]["payload"]


@pytest.mark.asyncio
async def test_agent_run_stream_emits_visible_thought_before_completion():
    db = make_test_session()
    user = _user(db, "visible-stream@example.com")

    event_types = []
    async for event in stream_agent_run(db, user.id, {"user_input": "hello"}):
        event_type = event["data"]["event_type"]
        event_types.append(event_type)
        if event_type == "run_completed":
            break

    assert "visible_thought" in event_types
    assert "run_completed" in event_types
    assert event_types.index("visible_thought") < event_types.index("run_completed")


def test_home_intent_react_detects_research_route():
    db = make_test_session()
    user = _user(db, "intent-research@example.com")

    result = run_agent(db, user.id, {"user_input": "帮我研究 LangGraph 多 Agent 趋势"})

    assert result["risk_level"] in {"L1", "L2"}
    assert "research_agent" in result["route_plan"]
    events = list_events(db, user.id, result["run_id"])
    assert any(event["data"]["node_name"] == "home_intent_react" for event in events)


def test_l3_email_task_creates_approval_and_events():
    db = make_test_session()
    user = _user(db, "approval-email@example.com")

    result = run_agent(db, user.id, {"user_input": "帮我给 Leo 发邮件说 demo 明天上午发"})

    assert result["status"] == "waiting_approval"
    assert result["approval_required"] is True
    assert result["risk_level"] == "L3"
    approvals = ApprovalRepository(db).list_by_user(user.id)
    assert approvals
    events = list_events(db, user.id, result["run_id"])
    assert any(event["event"] == "approval_required" for event in events)


def test_l4_delete_task_requires_strong_approval():
    db = make_test_session()
    user = _user(db, "approval-delete@example.com")

    result = run_agent(db, user.id, {"user_input": "帮我删除所有历史记录"})

    assert result["status"] == "waiting_approval"
    assert result["risk_level"] == "L4"
    assert result["approval_required"] is True


def test_approval_approve_and_reject_update_run_events():
    db = make_test_session()
    user = _user(db, "approval-update@example.com")
    result = run_agent(db, user.id, {"user_input": "send email to Leo"})
    approval = ApprovalRepository(db).list_by_user(user.id)[0]

    approved = update_approval_status(db, user.id, approval.id, "approved", {"decision": "approved"})

    assert approved["status"] == "approved"
    events = list_events(db, user.id, result["run_id"])
    assert any(event["event"] == "approval_approved" for event in events)
    assert any(event["event"] == "run_completed" for event in events)

    result2 = run_agent(db, user.id, {"user_input": "post comment to website"})
    approval2 = ApprovalRepository(db).list_by_user(user.id)[0]
    rejected = update_approval_status(db, user.id, approval2.id, "rejected", {"reason": "not now"})
    assert rejected["status"] == "rejected"
    events2 = list_events(db, user.id, result2["run_id"])
    assert any(event["event"] == "approval_rejected" for event in events2)


def test_checkpointer_falls_back_when_redis_unavailable():
    checkpointer = build_checkpointer("redis://127.0.0.1:0/0")
    assert checkpointer is not None
