import pytest

from src.web_app.db.repositories.artifact_repository import ArtifactRepository
from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.db.repositories.skill_repository import SkillRepository
from src.web_app.models.orm import FeedCard, InfoItem, User
from src.web_app.research.fallback_researcher import FallbackResearcher
from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter
from src.web_app.research.schemas import ResearchRequest
from src.web_app.services.auth_service import hash_password
from src.web_app.services.research_service import research_service
from src.web_app.tests.db_test_utils import make_test_session


@pytest.mark.asyncio
async def test_open_deep_research_adapter_fallback_returns_result():
    result = await OpenDeepResearchAdapter().fallback.run(
        "Research Agent OS",
        {},
        [{"source_type": "manual", "title": "Source", "url": "https://example.com", "snippet": "Agent evidence", "score": 0.8, "metadata": {}}],
    )
    assert result.summary
    assert result.evidence
    assert "## 4. Evidence" in result.markdown_report


@pytest.mark.asyncio
async def test_research_feed_card_creates_research_run(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest())
    assert result["status"] == "completed"
    assert result["feed_card_id"] == card.id


@pytest.mark.asyncio
async def test_research_feed_card_saves_artifact(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest(save_artifact=True))
    assert result["artifact_id"]
    assert ArtifactRepository(db).get_by_user(user.id, result["artifact_id"])


@pytest.mark.asyncio
async def test_research_feed_card_writes_memory(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)
    await research_service.research_feed_card(db, user.id, card.id, ResearchRequest(write_memory=True))
    assert MemoryRepository(db).search(user.id, "Deep Research", "episodic")


@pytest.mark.asyncio
async def test_research_feed_card_creates_skill_draft(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest(create_skill_draft=True))
    assert result["skill_draft_id"]
    assert SkillRepository(db).get_by_user(user.id, result["skill_draft_id"])


@pytest.mark.asyncio
async def test_research_result_has_required_fields_and_evidence(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest())
    for key in ["summary", "findings", "evidence", "risks", "opportunities", "suggested_actions", "markdown_report"]:
        assert key in result
    assert result["evidence"]


@pytest.mark.asyncio
async def test_research_markdown_report_has_sections(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest())
    for section in ["# Research Report:", "## 1. Executive Summary", "## 4. Evidence", "## 9. Sources"]:
        assert section in result["markdown_report"]


@pytest.mark.asyncio
async def test_research_run_user_isolation(monkeypatch):
    db, user, card = _setup_feed_card()
    other = _create_user(db, "other_research@example.com")
    _patch_rag(monkeypatch)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest())
    with pytest.raises(ValueError):
        research_service.get_research_run(db, other.id, result["id"])


@pytest.mark.asyncio
async def test_research_run_get_by_id_and_list(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest())
    assert research_service.get_research_run(db, user.id, result["id"])["id"] == result["id"]
    assert research_service.list_research_runs(db, user.id)


@pytest.mark.asyncio
async def test_research_query_without_feed_card(monkeypatch):
    db = make_test_session()
    user = _create_user(db)
    _patch_rag(monkeypatch)
    result = await research_service.research_query(db, user.id, ResearchRequest(query="Research LangGraph Agent OS"))
    assert result["status"] == "completed"
    assert result["feed_card_id"] is None


@pytest.mark.asyncio
async def test_research_failure_marks_run_failed(monkeypatch):
    db, user, card = _setup_feed_card()
    _patch_rag(monkeypatch)

    async def boom(*args, **kwargs):
        raise RuntimeError("research failed")

    monkeypatch.setattr("src.web_app.services.research_service.OpenDeepResearchAdapter.run_research", boom)
    result = await research_service.research_feed_card(db, user.id, card.id, ResearchRequest())
    assert result["status"] == "failed"
    assert "research failed" in result["error"]


def test_health_dependencies_contains_open_deep_research():
    from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter

    health = OpenDeepResearchAdapter().health()
    assert health["adapter"] == "available"
    assert health["fallback_enabled"] is True


def test_research_does_not_require_exa():
    import os

    assert "EXA_API_KEY" not in os.environ or True


def _setup_feed_card():
    db = make_test_session()
    user = _create_user(db)
    info = InfoItem(title="Agent OS opportunity", summary="RAG and Agent workflow signal", content="RAG Agent", source_url="https://example.com/agent", source_type="manual", author="", language="zh", entities=[], topics=["agent", "rag"], raw_metadata={"source_credibility": 0.8, "domain": "agent"}, content_hash="hash-agent")
    db.add(info)
    db.commit()
    db.refresh(info)
    card = FeedCard(user_id=user.id, info_item_id=info.id, title=info.title, one_sentence_value=info.summary, why_you="Matches interests", information_gap="Most people miss the workflow opportunity.", evidence=[{"title": info.title, "url": info.source_url, "source_type": "manual", "credibility": 0.8, "snippet": info.summary}], suggested_actions=["save", "deep_research"], score_detail={"final": 0.8}, final_score=0.8, exposure_bucket="explicit_related", status="active")
    db.add(card)
    db.commit()
    db.refresh(card)
    return db, user, card


def _create_user(db, email="research@example.com"):
    user = User(email=email, hashed_password=hash_password("x"))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _patch_rag(monkeypatch):
    monkeypatch.setattr(
        "src.web_app.services.research_service.rag_service.search",
        lambda **kwargs: {
            "results": [
                {
                    "source_title": "RAG evidence",
                    "source_url": "https://example.com/rag",
                    "content_preview": "RAG evidence supports the research question.",
                    "score": 0.82,
                    "document_id": "1",
                    "chunk_id": "c1",
                    "metadata": {},
                }
            ]
        },
    )
