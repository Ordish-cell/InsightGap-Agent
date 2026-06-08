import pytest

from src.web_app.agent.llm.config import clear_llm_settings_cache
from src.web_app.agent.llm.errors import LLMParseError, LLMUnavailableError
from src.web_app.agent.llm.factory import clear_chat_model_cache, get_chat_model
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.agent.runtime.intent_llm import infer_home_intent_with_llm
from src.web_app.agent.runtime.intent_schema import HomeIntentResult
from src.web_app.db.repositories.agent_repository import LLMCallRepository
from src.web_app.models.orm import AgentRun, FeedCard, InfoItem, User
from src.web_app.services.agent_service import list_events, run_agent
from src.web_app.services.feed_service import list_home_cards
from src.web_app.tests.db_test_utils import make_test_session


def _clear_llm():
    clear_llm_settings_cache()
    clear_chat_model_cache()


def _user(db, email="llm-stage11@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_model_router_resolves_aliyun_models(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "aliyun")
    # Pin all model env vars to known values so the local .env file
    # (which may set AGENT_INTENT_MODEL=qwen-turbo etc.) does not
    # override the field defaults under test.
    monkeypatch.setenv("AGENT_INTENT_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_SAFETY_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_MEMORY_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_SKILL_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_FAST_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_PLANNER_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_RAG_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_BALANCED_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_RESEARCH_MODEL", "qwen3.7-plus")
    monkeypatch.setenv("AGENT_STRONG_MODEL", "qwen3.7-plus")
    monkeypatch.setenv("AGENT_ARTIFACT_MODEL", "qwen3.6-plus")
    monkeypatch.setenv("AGENT_FINAL_MODEL", "qwen3.6-plus")
    monkeypatch.setenv("AGENT_LLM_MODEL", "qwen3.6-plus")
    monkeypatch.setenv("AGENT_EMBEDDING_PROVIDER", "aliyun")
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "text-embedding-v4")
    _clear_llm()

    assert resolve_model_name("intent").model == "qwen3.6-max-preview"
    assert resolve_model_name("safety").model == "qwen3.6-max-preview"
    assert resolve_model_name("planner").model == "qwen3.6-max-preview"
    assert resolve_model_name("research").model == "qwen3.7-plus"
    assert resolve_model_name("artifact").model == "qwen3.6-plus"
    assert resolve_model_name("rag").model == "qwen3.6-max-preview"
    assert resolve_model_name("final").model == "qwen3.6-plus"
    assert resolve_model_name("embedding").model == "text-embedding-v4"


def test_llm_factory_missing_api_key_does_not_crash_service(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_ENABLED", "true")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "aliyun")
    # Override .env file values — delenv only clears the OS env, but
    # pydantic-settings still reads the .env file.
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("ALIYUN_BAILIAN_API_KEY", "")
    _clear_llm()

    with pytest.raises(LLMUnavailableError):
        get_chat_model("intent")

    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {"user_input": "帮我研究 LangGraph 多 Agent 趋势"})
    assert result["status"] != "failed"
    assert result["intent"] == "research"
    assert result["risk_level"] == "L1"


def test_home_intent_llm_success_logs_usage(monkeypatch):
    class FakeModel:
        def invoke(self, _prompt):
            return type("Msg", (), {"content": '{"intent":"tool.email","confidence":0.92,"risk_level":"L3","needs_approval":true,"needs_clarification":false,"required_agents":["tool_agent","memory_agent"],"expected_output":"email_preview","reason_summary":"用户请求发邮件，需要审批。","suggested_route_hints":["tool_agent","memory_agent"],"tool_action_type":"email.send"}'})()

    monkeypatch.setattr("src.web_app.agent.runtime.intent_llm.get_chat_model", lambda *args, **kwargs: FakeModel())
    _clear_llm()
    db = make_test_session()
    user = _user(db, "intent-success@example.com")
    run = AgentRun(user_id=user.id, status="running", user_input="send email", graph_state={"thread_id": "t1"})
    db.add(run)
    db.commit()
    db.refresh(run)

    result = infer_home_intent_with_llm(db, run_id=run.id, thread_id="t1", user_id=user.id, user_input="send email", page_context={})

    assert result.intent == "tool.email"
    assert result.risk_level == "L3"
    assert LLMCallRepository(db).list_by_run(user.id, run.id)[0].status == "completed"
    assert any(event["event"] == "llm_call_completed" for event in list_events(db, user.id, run.id))


def test_home_intent_llm_invalid_json_falls_back_in_runtime(monkeypatch):
    class BadModel:
        def invoke(self, _prompt):
            return type("Msg", (), {"content": "not-json"})()

    monkeypatch.setattr("src.web_app.agent.runtime.intent_llm.get_chat_model", lambda *args, **kwargs: BadModel())
    _clear_llm()
    db = make_test_session()
    user = _user(db, "intent-invalid@example.com")

    result = run_agent(db, user.id, {"user_input": "帮我研究 LangGraph 多 Agent 趋势"})
    events = list_events(db, user.id, result["run_id"])

    assert result["intent"] == "research"
    assert any(event["event"] == "home_intent_fallback_used" for event in events)
    assert any(event["event"] == "llm_call_failed" for event in events)


def test_intent_schema_filters_unknown_agents():
    result = HomeIntentResult.model_validate({
        "intent": "research",
        "confidence": 0.8,
        "risk_level": "L1",
        "required_agents": ["research_agent", "evil_agent"],
        "suggested_route_hints": ["research_agent", "evil_agent"],
    })
    assert result.required_agents == ["research_agent"]
    assert result.suggested_route_hints == ["research_agent"]
    with pytest.raises(Exception):
        HomeIntentResult.model_validate({"intent": "research", "confidence": 2, "risk_level": "LOW"})


def test_langgraphstatus_contains_key_steps(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_ENABLED", "false")
    _clear_llm()
    db = make_test_session()
    user = _user(db, "status-stage11@example.com")

    result = run_agent(db, user.id, {"user_input": "你好"})
    status = result["langgraphstatus"]
    keys = {step["key"] for step in status["steps"]}

    assert {"home_intent", "planner", "final_response"} <= keys
    for step in status["steps"]:
        assert step.get("title")
        assert step.get("status")
        assert step.get("node_name")
        assert "detail" in step


def test_rag_embedding_model_resolution(monkeypatch):
    # Pin model env vars so .env overrides don't interfere with defaults
    monkeypatch.setenv("AGENT_RAG_MODEL", "qwen3.6-max-preview")
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "text-embedding-v4")
    _clear_llm()
    assert resolve_model_name("embedding").model == "text-embedding-v4"
    assert resolve_model_name("rag").model == "qwen3.6-max-preview"


def test_home_feed_three_real_cards_shape():
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
            )
        )
    db.commit()

    result = list_home_cards(db, user.id)

    assert result["is_complete"] is True
    assert len(result["cards"]) == 3
    for card in result["cards"]:
        assert card["title"]
        assert card["source_url"].startswith("https://")
        assert card["relation_type"]
        assert card["final_score"] > 0
        assert card["suggested_actions"]
