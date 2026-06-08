import pytest

from src.web_app.feed.card_generator import generate_feed_card
from src.web_app.feed.mixer import mix_cards
from src.web_app.feed.normalizer import normalize_raw_item
from src.web_app.feed.scorer import FeedScorer
from src.web_app.feed.sources.arxiv import ArxivSource
from src.web_app.feed.sources.base import RawFeedItem
from src.web_app.feed.sources.github import GitHubSource
from src.web_app.feed.sources.manager import SearchSourceManager
from src.web_app.feed.sources.manual_seed import ManualSeedSource
from src.web_app.feed.sources.rss import RSSSource
from src.web_app.feed.sources.serpapi import SerpApiSource
from src.web_app.feed.sources.tavily import TavilySource
from src.web_app.models.orm import User
from src.web_app.services.auth_service import hash_password
from src.web_app.services.feed_service import feedback, list_cards, refresh_feed, research_from_card, stats
from src.web_app.tests.db_test_utils import make_test_session


@pytest.mark.asyncio
async def test_manual_seed_source_fetch():
    rows = await ManualSeedSource().fetch()
    assert rows
    assert rows[0].url


@pytest.mark.asyncio
async def test_github_source_fetch_mocked(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"items": [{"full_name": "x/agent", "description": "agent repo", "html_url": "https://github.com/x/agent", "pushed_at": "2026-06-01T00:00:00Z", "owner": {"login": "x"}, "topics": ["agent"]}]}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs): return Response()
    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: Client())
    rows = await GitHubSource().fetch()
    assert rows[0].source_type == "github"


@pytest.mark.asyncio
async def test_arxiv_source_fetch_mocked(monkeypatch):
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1234.1</id><title>Agent Paper</title><summary>RAG agent</summary><published>2026-06-01T00:00:00Z</published><author><name>A</name></author><category term="cs.AI"/></entry></feed>"""
    class Response:
        text = xml
        def raise_for_status(self): pass
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs): return Response()
    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: Client())
    rows = await ArxivSource().fetch()
    assert rows[0].source_type == "paper"


def test_rss_source_fetch_mocked():
    rows = RSSSource()._parse("""<?xml version="1.0"?><rss><channel><title>Blog</title><item><title>Agent Blog</title><link>https://example.com/a</link><description>RAG</description></item></channel></rss>""", "local")
    assert rows[0].source_type == "blog"


def test_tavily_serpapi_disabled_without_key(monkeypatch):
    monkeypatch.setattr("src.web_app.feed.sources.tavily.settings.tavily_api_key", "")
    monkeypatch.setattr("src.web_app.feed.sources.serpapi.settings.serpapi_api_key", "")
    assert TavilySource().health()["status"] == "disabled"
    assert SerpApiSource().health()["status"] == "disabled"


@pytest.mark.asyncio
async def test_search_source_manager_fetch_all_partial_failure():
    class BadSource:
        name = "bad"
        enabled = True
        async def fetch(self): raise RuntimeError("boom")
        def health(self): return {"enabled": True}
    manager = SearchSourceManager([BadSource(), ManualSeedSource()])
    rows, source_stats = await manager.fetch_all()
    assert rows
    assert source_stats["bad"]["status"] == "degraded"


def test_search_sources_do_not_require_exa():
    names = SearchSourceManager().health().keys()
    assert not any("exa" in name.lower() for name in names)


def test_feed_score_formula_and_low_confidence():
    db = make_test_session()
    user = _user(db)
    profile = user.profile
    raw = RawFeedItem("m1", "manual", "Agent RAG MCP", "agent rag opportunity", "https://example.com", tags=["agent", "rag"])
    item = normalize_raw_item(raw)
    info = type("Info", (), item.__dict__)()
    score = FeedScorer().score(info, profile)
    assert score["final"] > 0
    raw_no_url = RawFeedItem("m2", "unknown", "Unknown", "tiny", None)
    item2 = normalize_raw_item(raw_no_url)
    info2 = type("Info", (), item2.__dict__)()
    assert FeedScorer().score(info2, profile)["confidence"] == "low"


def test_feed_mixer_ratio_30_40_30():
    cards = [{"relation_type": relation, "final_score": 0.9 - i * 0.01, "domain": f"d{i}", "source_type": f"s{i}", "confidence": "high"} for i, relation in enumerate(["explicit_related"] * 5 + ["adjacent_domain"] * 5 + ["far_domain"] * 5)]
    mixed, bucket_info = mix_cards(cards, {"explicit_related": 0.3, "adjacent_domain": 0.4, "far_domain": 0.3}, 10)
    assert len(mixed) == 10


def test_feed_refresh_creates_info_items_and_cards():
    db = make_test_session()
    user = _user(db)
    result = refresh_feed(db, user.id)
    assert result["created_info_items"] >= 1
    assert result["created_feed_cards"] >= 1
    cards = list_cards(db, user.id)["cards"]
    card = cards[0]
    for key in ["title", "one_sentence_value", "why_you", "information_gap", "evidence", "suggested_actions", "score", "relation_type", "source_type", "final_score"]:
        assert key in card


def test_feed_cards_user_isolation_and_feedback():
    db = make_test_session()
    user = _user(db, "u1@example.com")
    other = _user(db, "u2@example.com")
    refresh_feed(db, user.id)
    card = list_cards(db, user.id)["cards"][0]
    assert list_cards(db, other.id)["cards"] == []
    assert feedback(db, user.id, int(card["id"]), "save")["action"] == "save"
    feedback(db, user.id, int(card["id"]), "ignore")
    assert all(item["id"] != card["id"] for item in list_cards(db, user.id)["cards"])


def test_feed_research_and_stats():
    db = make_test_session()
    user = _user(db)
    refresh_feed(db, user.id)
    card = list_cards(db, user.id)["cards"][0]
    research = research_from_card(db, user.id, int(card["id"]))
    assert research["status"] == "not_implemented"
    assert stats(db, user.id)["cards_count"] >= 1


def _user(db, email="feed@example.com"):
    from src.web_app.db.repositories.profile_repository import ProfileRepository

    user = User(email=email, hashed_password=hash_password("x"))
    db.add(user)
    db.commit()
    db.refresh(user)
    profile = ProfileRepository(db).get_or_create_default(user.id)
    profile.explicit_interests = ["agent", "rag", "mcp", "langgraph"]
    profile.adjacent_domains = ["automation", "browser agent"]
    profile.far_domains = ["startup", "investment"]
    db.commit()
    user.profile = profile
    return user
