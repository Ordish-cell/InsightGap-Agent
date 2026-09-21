"""Grounded copy, user isolation, and production Feed generation paths."""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src.web_app.feed import card_generator as generator
from src.web_app.feed.personalization import NO_PERSONALIZATION


def profile(**kwargs):
    return SimpleNamespace(**{
        "user_id": 1, "explicit_interests": [], "goals": [], "adjacent_domains": [],
        "far_domains": [], "disliked_topics": [], **kwargs,
    })


def memory(**kwargs):
    return {"id": 7, "user_id": 1, "memory_type": "semantic", "content": "使用 Qdrant 检索", "metadata": {}, **kwargs}


def card(user=None, memories=None, *, title="Qdrant 与教育研究", summary="检索教学资料", tags=None, domain="rag", relation="explicit_related"):
    item = SimpleNamespace(title=title, summary=summary, topics=tags or [], raw_metadata={"domain": domain, "tags": tags or []}, source_type="web", source_url="https://example.com/signal", published_at=None)
    score = {"relation_type": relation, "source_credibility": 0.8, "final": 0.8, "confidence": "high"}
    return generator.generate_feed_card(item, score, user or profile(), memories)


def test_same_signal_different_users_and_evidence():
    first = card(profile(explicit_interests=["Qdrant"]))
    second = card(profile(user_id=2, explicit_interests=["教育研究"]))
    assert "Qdrant" in first["why_you"] and "教育研究" not in first["why_you"]
    assert "教育研究" in second["why_you"] and "Qdrant" not in second["why_you"]
    assert first["why_you"] == first["why_relevant"]
    assert first["personalization_evidence"] == [{"source_type": "profile", "profile_field": "explicit_interests", "text": "Qdrant", "matched_terms": ["qdrant"]}]


@pytest.mark.parametrize("field", ["explicit_interests", "goals", "adjacent_domains", "far_domains"])
def test_all_profile_fields_are_attributed(field):
    result = card(profile(**{field: ["教育研究"]}))
    assert result["personalization_evidence"][0]["profile_field"] == field


@pytest.mark.parametrize("relation", ["explicit_related", "adjacent_domain", "far_domain"])
def test_no_facts_or_no_match_never_invents_relevance(relation):
    for user in (profile(), profile(explicit_interests=["农业"]), profile(explicit_interests=["育"])):
        result = card(user, relation=relation)
        assert result["why_you"] == result["why_relevant"] == NO_PERSONALIZATION
        assert result["personalization_evidence"] == []


def test_memory_reference_and_profile_priority_dedup_and_limit():
    result = card(profile(explicit_interests=["Qdrant"], goals=["QDRANT"]), [memory(content="qdrant"), memory(id=8, content="教育研究"), memory(id=9, content="检索教学资料")])
    evidence = result["personalization_evidence"]
    assert len(evidence) == 2
    assert evidence[0]["source_type"] == "profile"
    assert evidence[1]["memory_id"] == 8
    assert "已保存记忆提到『教育研究』" in result["why_you"]
    assert result == card(profile(explicit_interests=["Qdrant"], goals=["QDRANT"]), [memory(id=9, content="检索教学资料"), memory(id=8, content="教育研究"), memory(content="qdrant")])


@pytest.mark.parametrize("override", [
    {"user_id": 2}, {"user_id": None}, {"id": None}, {"memory_type": "episodic"},
    {"metadata": {"status": "archived"}}, {"metadata": {"status": "superseded"}},
    {"metadata": {"status": "deleting"}}, {"metadata": {"sensitive": True}},
    {"metadata": {"confidence": 0.3}}, {"metadata": {"confidence": "invalid"}},
    {"metadata": {"confidence": float("nan")}}, {"_low_confidence": True},
    {"metadata": {"category": "negative_preference"}}, {"content": "不喜欢 Qdrant"},
    {"content": "I don't use Qdrant"}, {"content": "I do not like Qdrant"},
])
def test_ineligible_memory_is_not_personalization(override):
    assert card(memories=[memory(**override)])["why_you"] == NO_PERSONALIZATION


def test_unknown_owner_is_not_allowed_to_use_memory():
    assert card(profile(user_id=None), [memory()])["why_you"] == NO_PERSONALIZATION


def test_negative_profile_and_disliked_terms():
    assert card(profile(explicit_interests=["不喜欢 Qdrant"]))["why_you"] == NO_PERSONALIZATION
    assert card(profile(disliked_topics=["Qdrant"]), [memory()])["why_you"] == NO_PERSONALIZATION


def test_word_boundaries_normalization_and_chinese_tag_matching():
    assert card(profile(explicit_interests=["RAG"]), title="storage", summary="", tags=[])["why_you"] == NO_PERSONALIZATION
    result = card(profile(goals=["研究 教育产品 的需求"]), title="教育产品研究", tags=["教育产品"])
    assert result["personalization_evidence"][0]["matched_terms"] == ["教育产品"]
    assert card(profile(explicit_interests=["  ＱＤＲＡＮＴ  "]))["personalization_evidence"][0]["matched_terms"] == ["qdrant"]
    assert card(profile(explicit_interests=["研究"]), title="研究员", summary="")["why_you"] == NO_PERSONALIZATION
    assert card(profile(explicit_interests=["教育"]), title="AI教育", summary="")["personalization_evidence"][0]["matched_terms"] == ["教育"]


def test_quality_fallback_keeps_aliases_and_grounding():
    result = generator.validate_card_quality({"why_you": "你的 Agent OS", "domain": "ai"})
    assert result["why_you"] == result["why_relevant"] == NO_PERSONALIZATION
    assert result["next_action"]
    grounded = card(profile(explicit_interests=["Qdrant"]))
    grounded["why_relevant"] = ""
    repaired = generator.validate_card_quality(grounded)
    assert repaired["why_you"] == repaired["why_relevant"]
    assert "Qdrant" in repaired["why_you"]


def test_every_template_and_domain_remains_signal_only():
    templates = [*generator._NEXT_ACTIONS]
    for mapping in (generator._BENEFIT_TEMPLATES, generator._GAP_TEMPLATES, generator._VALUE_PREFIXES):
        templates.extend(t for variants in mapping.values() for t in variants)
    assert all(not any(term in t for term in ("你的", "你正在", "你当前", "Agent OS", "FastAPI", "Qdrant")) for t in templates)
    for domain in (*generator._DOMAIN_CN, "unknown"):
        for relation in ("explicit_related", "adjacent_domain", "far_domain"):
            result = card(title="中性来源", summary="原始摘要", domain=domain, relation=relation)
            for field in ("title", "summary", "one_sentence_value", "benefit", "information_gap", "next_action"):
                assert not any(term in result[field] for term in ("你的", "你正在", "你当前", "Agent OS", "FastAPI", "Qdrant"))


def test_real_signal_technology_names_are_preserved():
    result = card(title="FastAPI 与 Qdrant", summary="来源讨论 FastAPI 和 Qdrant", tags=["FastAPI", "Qdrant"])
    assert result["title"] == "FastAPI 与 Qdrant"
    assert "Qdrant" in result["summary"]
    assert result["why_you"] == NO_PERSONALIZATION


@pytest.mark.parametrize("seed_only", [False, True])
@pytest.mark.parametrize("memory_failure", [False, True])
@pytest.mark.parametrize("with_profile", [False, True])
def test_refresh_persists_grounding_without_mutating_profile(monkeypatch, seed_only, memory_failure, with_profile):
    from src.web_app.db.repositories.profile_repository import ProfileRepository
    from src.web_app.feed.sources.base import RawFeedItem
    from src.web_app.feed.sources.manager import SearchSourceManager
    from src.web_app.models.orm import FeedCard, Memory, User
    from src.web_app.services import feed_service as service
    from src.web_app.tests.db_test_utils import make_test_session

    db = make_test_session()
    user = User(email="grounded@example.com", hashed_password="unused")
    other = User(email="other@example.com", hashed_password="unused")
    db.add_all([user, other])
    db.commit()
    own_profile = ProfileRepository(db).get_or_create_default(user.id)
    if with_profile:
        own_profile.explicit_interests = ["LangGraph"]
    own_memory = Memory(user_id=user.id, content="LangGraph", memory_type="semantic", importance=0.9, metadata_json={"status": "active"})
    db.add_all([own_memory, Memory(user_id=other.id, content="Qdrant", memory_type="semantic", importance=0.9, metadata_json={})])
    db.commit()

    async def fetch(_self):
        items = [] if seed_only else [RawFeedItem(source_id="grounded-guide", source_type="github", title="LangGraph retrieval guide", url="https://example.com/guide", summary="LangGraph guide", tags=["langgraph"])]
        return items, {}, {}

    monkeypatch.setattr(SearchSourceManager, "fetch_all", fetch)
    if memory_failure:
        def unavailable(*args, **kwargs):
            raise RuntimeError("memory unavailable")
        monkeypatch.setattr(service.memory_service, "get_semantic_memories", unavailable)
    result = service.refresh_feed(db, user.id)
    assert result["created_feed_cards"] > 0
    rows = list(db.scalars(select(FeedCard).where(FeedCard.user_id == user.id)))
    assert rows
    grounded = []
    for row in rows:
        evidence = row.score_detail["personalization_evidence"]
        assert row.why_you == row.score_detail["why_relevant"]
        if evidence:
            grounded.append(row)
            assert all(e["source_type"] == "profile" or e["memory_id"] == own_memory.id for e in evidence)
            if with_profile:
                assert evidence[0]["source_type"] == "profile"
        else:
            assert row.why_you == NO_PERSONALIZATION
    if memory_failure and not with_profile:
        assert not grounded
    else:
        assert grounded
        if not seed_only:
            assert any(r.score_detail["original_title"] == "LangGraph retrieval guide" for r in grounded)
        assert any(r.score_detail["source_kind"] == "bucket_seed" for r in grounded)
    service.list_cards(db, user.id)
    db.commit()
    db.refresh(own_profile)
    assert own_profile.explicit_interests == (["LangGraph"] if with_profile else [])
    assert all(getattr(own_profile, field) == [] for field in ("goals", "adjacent_domains", "far_domains"))
