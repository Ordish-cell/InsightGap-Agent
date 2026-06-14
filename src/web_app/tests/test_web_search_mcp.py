from src.web_app.core.constants import L0_READ_ONLY
from src.web_app.db.repositories.mcp_repository import ToolCallRepository
from src.web_app.mcp.web_search_provider import WebSearchProvider, web_search_provider
from src.web_app.models.orm import User
from src.web_app.services.mcp_service import mcp_service
from src.web_app.tests.db_test_utils import make_test_session


def _user(db, email: str = "web-search@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_web_search_registry_seeds_readonly_tool():
    db = make_test_session()

    tools = {tool["name"]: tool for tool in mcp_service.list_tools(db)}

    assert "web.search" in tools
    assert tools["web.search"]["safety_level"] == L0_READ_ONLY
    assert tools["web.search"]["requires_approval"] is False
    assert tools["web.search"]["enabled"] is True


def test_web_search_provider_prefers_tavily(monkeypatch):
    provider = WebSearchProvider()
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.tavily_api_key", "tavily-key")
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.serpapi_api_key", "serpapi-key")
    monkeypatch.setattr(provider, "_search_tavily", lambda query, limit, recency_days: {
        "query": query,
        "provider": "tavily",
        "used_fallback": False,
        "results": [{"title": "A", "url": "https://a.example", "snippet": "A", "provider": "tavily"}],
        "error": "",
    })
    monkeypatch.setattr(provider, "_search_serpapi", lambda query, limit, recency_days: {
        "query": query,
        "provider": "serpapi",
        "used_fallback": True,
        "results": [{"title": "B", "url": "https://b.example", "snippet": "B", "provider": "serpapi"}],
        "error": "",
    })

    result = provider.search("OpenAI latest", limit=3)

    assert result["provider"] == "tavily"
    assert result["results"][0]["url"] == "https://a.example"


def test_web_search_provider_falls_back_to_serpapi(monkeypatch):
    provider = WebSearchProvider()
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.tavily_api_key", "tavily-key")
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.serpapi_api_key", "serpapi-key")
    monkeypatch.setattr(provider, "_search_tavily", lambda query, limit, recency_days: {
        "query": query,
        "provider": "tavily",
        "used_fallback": False,
        "results": [],
        "error": "boom",
    })
    monkeypatch.setattr(provider, "_search_serpapi", lambda query, limit, recency_days: {
        "query": query,
        "provider": "serpapi",
        "used_fallback": False,
        "results": [{"title": "B", "url": "https://b.example", "snippet": "B", "provider": "serpapi"}],
        "error": "",
    })

    result = provider.search("OpenAI latest", limit=3)

    assert result["provider"] == "serpapi"
    assert result["used_fallback"] is True


def test_web_search_provider_returns_structured_failure(monkeypatch):
    provider = WebSearchProvider()
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.tavily_api_key", "")
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.serpapi_api_key", "")

    result = provider.search("OpenAI latest", limit=3)

    assert result["results"] == []
    assert result["error"] == "no_provider_configured"
    assert result["search_rounds"]


def test_web_search_provider_runs_second_round_when_first_round_is_weak(monkeypatch):
    provider = WebSearchProvider()
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.tavily_api_key", "tavily-key")
    monkeypatch.setattr("src.web_app.mcp.web_search_provider.settings.serpapi_api_key", "")

    calls = []

    def fake_tavily(query, limit, recency_days):
        calls.append(query)
        if query == "Gemini 的最新模型是什么？":
            return {
                "query": query,
                "provider": "tavily",
                "used_fallback": False,
                "results": [{"title": "Gemini - YouTube", "url": "https://youtube.com/gemini", "snippet": "", "provider": "tavily"}],
                "error": "",
            }
        return {
            "query": query,
            "provider": "tavily",
            "used_fallback": False,
            "results": [{"title": "Gemini official models", "url": "https://ai.google.dev/gemini-api/docs/models", "snippet": "Official models", "provider": "tavily"}],
            "error": "",
        }

    monkeypatch.setattr(provider, "_search_tavily", fake_tavily)

    result = provider.search("Gemini 的最新模型是什么？", limit=3)

    assert len(calls) == 2
    assert result["results"][0]["url"] == "https://ai.google.dev/gemini-api/docs/models"
    assert len(result["search_rounds"]) == 2
    assert "读取到" in result["reasoning_summary"]


def test_web_search_tool_records_tool_call(monkeypatch):
    db = make_test_session()
    user = _user(db)
    monkeypatch.setattr(web_search_provider, "search", lambda query, limit=5, recency_days=None: {
        "query": query,
        "provider": "tavily",
        "used_fallback": False,
        "results": [{"title": "OpenAI", "url": "https://openai.com", "snippet": "News", "provider": "tavily"}],
        "error": "",
    })

    result = mcp_service.call_tool(db, user.id, "web.search", {"query": "OpenAI latest", "limit": 2})

    assert result["status"] == "completed"
    assert result["output"]["results"][0]["url"] == "https://openai.com"
    calls = ToolCallRepository(db).list_by_user(user.id)
    assert calls[0].tool_name == "web.search"
