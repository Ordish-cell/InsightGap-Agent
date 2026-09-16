import pytest

pytestmark = pytest.mark.usefixtures("scripted_supervisor")

from src.web_app.models.orm import FeedCard, InfoItem, User
from src.web_app.services.rag_service import rag_service
from src.web_app.services import research_service as research_service_module
from src.web_app.services import agent_service
from src.web_app.services.agent_service import list_steps, run_agent
from src.web_app.tests.db_test_utils import make_test_session, configure_test_model


def _user(db, email="stage6@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    configure_test_model(db, user)
    return user


def _feed_card(db, user_id: int) -> FeedCard:
    item = InfoItem(title="LangGraph Runtime", summary="Agent orchestration", content="LangGraph connects runtime nodes.", source_url="https://example.com/runtime")
    db.add(item)
    db.commit()
    db.refresh(item)
    card = FeedCard(
        user_id=user_id,
        info_item_id=item.id,
        title="LangGraph Agent Runtime opportunity",
        one_sentence_value="Unify feed, research, memory, and artifacts.",
        why_you="Matches the user's Agent OS direction.",
        information_gap="Most demos stop before durable runtime orchestration.",
        evidence=[{"title": "Runtime note", "url": item.source_url, "summary": item.summary}],
        suggested_actions=["Run deep research"],
        score_detail={},
        final_score=0.9,
        exposure_bucket="explicit_related",
    )
    db.add(card)
    db.commit()
    db.refresh(card)
    return card


def test_agent_runtime_routes_to_rag(monkeypatch):
    db = make_test_session()
    user = _user(db)

    monkeypatch.setattr(rag_service, "ask", lambda *args, **kwargs: {"answer": "RAG answer", "retrieval_status": "ok", "evidence": [{"evidence_id": "E1", "quote": "RAG answer"}]})

    result = run_agent(db, user.id, {"user_input": "请基于知识库资料回答", "route": "rag"})

    assert result["status"] == "completed"
    assert result["route"] == "rag"
    assert result["rag"]["evidence"]
    assert "RAG answer" in str(result["rag"]["evidence"])
    step_names = {step["node_name"] for step in list_steps(db, user.id, result["run_id"])}
    assert step_names == {"supervisor"}
    assert any(step["output"].get("capability") == "rag" for step in list_steps(db, user.id, result["run_id"]))


@pytest.mark.asyncio
async def test_agent_runtime_research_feed_card_creates_outputs(monkeypatch):
    db = make_test_session()
    user = _user(db, "research-stage6@example.com")
    card = _feed_card(db, user.id)
    monkeypatch.setattr(research_service_module.rag_service, "search", lambda *args, **kwargs: {"results": []})

    result = await agent_service.run_agent_async(db, user.id, {"user_input": "深度研究这张信息差卡片", "route": "research", "feed_card_id": card.id, "save_artifact": True, "create_skill_draft": True})

    assert result["status"] == "completed"
    assert result["route"] in ("research", "feed_research")
    assert result["research"]["status"] == "completed"
    assert result["artifacts"]
    assert result["skill_drafts"]
    from src.web_app.models.orm import Memory
    from sqlalchemy import select
    assert not list(db.scalars(select(Memory).where(Memory.user_id == user.id)))
    assert result["research"]["agent_run_id"] == result["run_id"]


def test_agent_runtime_external_write_waits_for_approval():
    db = make_test_session()
    user = _user(db, "approval-stage6@example.com")

    result = run_agent(db, user.id, {"user_input": "帮我发送邮件给客户", "route": "tool", "tool_name": "email.send", "tool_input": {"to": "client@example.test", "subject": "demo", "body": "demo"}})

    assert result["status"] == "waiting_approval"
    assert result["route"] in {"tool", "tool.email", "approval"}
    assert result["approval_required"] is True
    assert result["approval_payload"]["tool_name"] == "email.send"


def test_agent_runtime_high_risk_is_denied():
    db = make_test_session()
    user = _user(db, "blocked-stage6@example.com")

    # Even an administrator-enabled L4 tool remains blocked by execution policy.
    from src.web_app.services.mcp_service import mcp_service
    from src.web_app.db.repositories.mcp_repository import MCPToolRepository
    mcp_service.ensure_builtin_tools(db)
    tool = MCPToolRepository(db).get_by_name("local_file.delete")
    tool.enabled = True
    db.commit()

    result = run_agent(db, user.id, {"user_input": "Delete a file", "tool_name": "local_file.delete", "tool_input": {"path": "protected.txt"}})

    assert result["status"] == "completed"
    steps = list_steps(db, user.id, result["run_id"])
    assert any(s["output"].get("capability") == "tool" and s["output"]["status"] == "blocked" for s in steps)
    assert result["approval_required"] is False
    assert result["risk_level"] == "L4"


def test_agent_runtime_events_from_steps(monkeypatch):
    db = make_test_session()
    user = _user(db, "events-stage6@example.com")
    monkeypatch.setattr(rag_service, "ask", lambda *args, **kwargs: {"answer": "", "retrieval_status": "empty", "evidence": []})

    result = run_agent(db, user.id, {"user_input": "知识库问答", "route": "rag"})
    events = agent_service.replay_events(db, user.id, result["run_id"], limit=500)["events"]

    assert events
    assert events[0]["event_type"] == "run_created"
    assert any(event["event_type"] == "run_started" for event in events)
    assert any(event["event_type"] == "capability_result" and event["payload"]["capability"] == "rag" and event["payload"]["status"] == "empty" for event in events)


def test_health_dependencies_contains_agent_runtime():
    from src.web_app.api.v1.health import dependencies

    data = dependencies()["data"]
    assert data["agent_runtime"]["status"] == "ok"
    assert data["agent_runtime"]["fallback_enabled"] is False
