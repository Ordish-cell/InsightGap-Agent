import pytest

from src.web_app.agent.llm.config import clear_llm_settings_cache
from src.web_app.agent.llm.errors import LLMParseError, LLMUnavailableError
from src.web_app.agent.llm.factory import clear_chat_model_cache, get_chat_model
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.agent.runtime.nodes import SupervisorNodes
from pydantic import ValidationError
from src.web_app.db.repositories.agent_repository import LLMCallRepository
from src.web_app.models.orm import AgentRun, FeedCard, InfoItem, User
from src.web_app.services.agent_service import replay_events, run_agent
from src.web_app.services.feed_service import list_home_cards
from src.web_app.tests.db_test_utils import make_test_session, configure_test_model


def _clear_llm():
    clear_llm_settings_cache()
    clear_chat_model_cache()


def _user(db, email="llm-stage11@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    configure_test_model(db, user)
    return user


def test_model_router_uses_selected_run_model(monkeypatch, selected_test_model):
    monkeypatch.setenv("AGENT_RESEARCH_MODEL", "ignored-legacy-model")
    for purpose in ("intent", "safety", "supervisor", "research", "artifact", "rag", "final"):
        assert resolve_model_name(purpose).model == selected_test_model.model


def test_missing_model_setup_fails_explicitly():
    from src.web_app.services.agent_service import prepare_agent_run
    from src.web_app.services.llm_registry_service import ModelSetupError
    with pytest.raises(LLMUnavailableError, match="model_setup_required"):
        get_chat_model("intent")
    db = make_test_session()
    user = User(email="unconfigured@example.test", hashed_password="x")
    db.add(user)
    db.commit()
    with pytest.raises(ModelSetupError, match="model_setup_required"):
        prepare_agent_run(db, user.id, {"user_input": "research"})


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_supervisor_decision_logs_usage(monkeypatch, selected_test_model, valid):
    from types import SimpleNamespace
    from langchain_core.messages import AIMessageChunk
    from src.web_app.agent.llm.native_turn import NativeProtocolError
    async def invoke(prompt):
        yield AIMessageChunk(content="Hello") if valid else AIMessageChunk(content="", invalid_tool_calls=[{"id": "bad", "name": "unknown", "args": "{"}])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: SimpleNamespace(astream=invoke, bind_tools=lambda tools: SimpleNamespace(astream=invoke)))
    db = make_test_session()
    user = _user(db)
    run = AgentRun(user_id=user.id, status="running", user_input="hello")
    db.add(run)
    db.commit()
    nodes = SupervisorNodes(db, {})
    state = {"run_id": run.id, "user_id": user.id, "user_input": "hello"}
    await nodes.permission_guard(state)
    if valid:
        assert (await nodes.model_turn(state))[0].tool_call is None
    else:
        with pytest.raises(NativeProtocolError):
            await nodes.model_turn(state)
    logs = LLMCallRepository(db).list_by_run(user.id, run.id)
    assert logs[0].status == ("completed" if valid else "failed")


def test_chat_service_streams_without_legacy_steps(monkeypatch):
    from src.web_app.tests.test_chat_fast_path import fake_model
    fake_model(monkeypatch, ["你好"])
    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {"user_input": "你好"})
    assert result["status"] == "completed"
    events = replay_events(db, user.id, result["run_id"], limit=500)["events"]
    assert any(e["event_type"] == "answer_completed" for e in events)
    assert any(e["node_name"] == "supervisor" for e in events)
    assert not any(e["node_name"] in {"chat_entry", "home_intent_react"} for e in events)


def test_rag_embedding_model_resolution(monkeypatch, selected_test_model):
    # Pin model env vars so .env overrides don't interfere with defaults
    monkeypatch.setenv("AGENT_RAG_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "text-embedding-v4")
    _clear_llm()
    assert resolve_model_name("embedding").model == "text-embedding-v4"
    assert resolve_model_name("rag").model == selected_test_model.model


def test_home_feed_three_real_cards_shape(monkeypatch):
    monkeypatch.setattr("src.web_app.services.feed_service.maybe_refresh_for_user", lambda *a, **kw: {"refreshed": False})
    db = make_test_session()
    user = _user(db, "home-feed@example.com")
    for index in range(3):
        item = InfoItem(
            title=f"Real source {index}",
            summary="real summary",
            content="real content",
            source_url=f"https://github.com/example/repo-{index}",
            source_type="github",
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        db.add(
            FeedCard(
                user_id=user.id,
                info_item_id=item.id,
                title=item.title,
                one_sentence_value=item.summary,
                why_you="relevant",
                information_gap="gap",
                evidence=[{"url": item.source_url, "title": item.title}],
                suggested_actions=["带入对话"],
                score_detail={"source_type": "github", "source_name": "GitHub", "summary": item.summary},
                final_score=0.9 - index * 0.1,
                exposure_bucket=["explicit_related", "adjacent_domain", "far_domain"][index],
                status="active",
                batch_id="test-batch",
            )
        )
    db.commit()

    result = list_home_cards(db, user.id)

    assert result["is_complete"] is False  # Three cards do not satisfy the five-card batch target.
    assert len(result["cards"]) == 3
    for card in result["cards"]:
        assert card["title"]
        assert card["source_url"].startswith("https://")
        assert card["relation_type"]
        assert card["final_score"] > 0
        assert card["suggested_actions"]
