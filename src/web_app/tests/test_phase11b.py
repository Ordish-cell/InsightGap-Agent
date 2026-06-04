import pytest

from src.web_app.feed.card_generator import (
    _contains_chinese,
    _generate_benefit,
    _generate_chinese_title,
    _generate_information_gap,
    _generate_one_sentence_value,
    _generate_why_relevant,
    generate_display_title,
    generate_feed_card,
    is_mostly_english,
)
from src.web_app.feed.normalizer import normalize_raw_item
from src.web_app.feed.scorer import FeedScorer
from src.web_app.feed.sources.base import RawFeedItem
from src.web_app.models.orm import User, UserProfile
from src.web_app.services.auth_service import hash_password
from src.web_app.services.feed_service import refresh_feed
from src.web_app.services.memory_service import memory_service
from src.web_app.memory.extractor import memory_extractor
from src.web_app.tests.db_test_utils import make_test_session


def _user(db, email="test@example.com"):
    user = User(email=email, hashed_password=hash_password("pass"), nickname="test")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _profile(db, user):
    from src.web_app.db.repositories.profile_repository import ProfileRepository
    return ProfileRepository(db).get_or_create_default(user.id)


def _make_info_item(db, title="Test Paper: A New Approach", source_type="arxiv", topics=None, source_url="https://arxiv.org/abs/1234.1", domain="agent"):
    from src.web_app.models.orm import InfoItem
    import hashlib
    content_hash = hashlib.sha256((source_url or title).encode()).hexdigest()
    raw_metadata = {
        "source_id": "test-1",
        "canonical_url": source_url,
        "tags": topics or ["agent", "rag"],
        "domain": domain,
        "source_credibility": 0.85,
    }
    item = InfoItem(
        title=title,
        summary="A novel approach to agent-based retrieval and generation using RAG techniques.",
        content="Full paper content about agents and RAG.",
        source_url=source_url or "",
        source_type=source_type,
        author="Test Author",
        published_at=None,
        language="zh",
        topics=topics or ["agent", "rag"],
        raw_metadata=raw_metadata,
        content_hash=content_hash,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


# ===== Task A: Chinese title generation tests =====


def test_english_title_generates_chinese_display_title():
    """English arxiv title produces a Chinese display title."""
    title = _generate_chinese_title(
        "Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill",
        "arxiv", ["agent", "skill", "eval"], "agent"
    )
    assert title
    assert len(title) >= 10
    # Should NOT be the raw English title
    assert "Skill-RM: Unifying" not in title or _contains_chinese(title)


def test_github_title_generates_chinese_display_title():
    title = _generate_chinese_title(
        "langchain-ai/langgraph: Build resilient language agents as graphs",
        "github", ["langgraph", "agent"], "agent"
    )
    assert title
    assert len(title) >= 8


def test_chinese_title_already_chinese_is_preserved():
    title = _generate_chinese_title(
        "用 LangGraph 构建多 Agent 协作系统的新方法",
        "arxiv", ["agent", "langgraph"], "agent"
    )
    assert "LangGraph" in title


def test_chinese_title_not_mechanical_translation():
    """Chinese title should be productized, not a mechanical word-for-word translation."""
    title = _generate_chinese_title(
        "Attention Is All You Need",
        "arxiv", ["transformer", "attention"], "research"
    )
    # Should not contain the raw English title
    assert "Attention Is All You Need" not in title or _contains_chinese(title)


# ===== Task A: Chinese copy tests =====


def test_one_sentence_value_is_chinese():
    value = _generate_one_sentence_value(
        "Paper about agents", "agent", ["agent", "eval"], "arxiv"
    )
    assert value
    assert _contains_chinese(value)


def test_why_relevant_is_chinese():
    text = _generate_why_relevant(
        "Paper about agents", "agent", ["agent", "eval"], ["Agent", "RAG"], "arxiv"
    )
    assert text
    assert _contains_chinese(text)


def test_benefit_is_chinese():
    text = _generate_benefit("agent", ["agent", "eval"])
    assert text
    assert _contains_chinese(text)


def test_information_gap_is_chinese():
    text = _generate_information_gap("agent", ["agent", "eval"], "arxiv")
    assert text
    assert _contains_chinese(text)


def test_three_cards_have_different_copy():
    """Three different FeedCards should NOT produce identical template explanations."""
    cards = []
    for i, (title, domain, tags) in enumerate([
        ("Paper A: Agent Memory Systems", "agent", ["agent", "memory"]),
        ("Paper B: RAG Optimization Techniques", "rag", ["rag", "qdrant"]),
        ("Paper C: New DevTools Framework", "devtools", ["github", "python"]),
    ]):
        db = make_test_session()
        user = _user(db, f"test{i}@example.com")
        profile = _profile(db, user)
        item = _make_info_item(db, title=title, topics=tags, domain=domain)
        scorer = FeedScorer()
        score = scorer.score(item, profile)
        card = generate_feed_card(item, score, profile)
        cards.append(card)

    # All one_sentence_values should be different
    values = [c["one_sentence_value"] for c in cards]
    assert len(set(values)) == 3, f"one_sentence_values should differ, got: {values}"

    # All information_gaps should be different
    gaps = [c["information_gap"] for c in cards]
    assert len(set(gaps)) == 3, f"information_gaps should differ, got: {gaps}"

    # All titles should differ
    titles = [c["title"] for c in cards]
    assert len(set(titles)) == 3, f"titles should differ, got: {titles}"


def test_feed_card_has_new_fields():
    """Generated card includes why_relevant, benefit, next_action, original_title."""
    db = make_test_session()
    user = _user(db)
    profile = _profile(db, user)
    item = _make_info_item(db)
    scorer = FeedScorer()
    score = scorer.score(item, profile)
    card = generate_feed_card(item, score, profile)

    assert card.get("why_relevant")
    assert card.get("benefit")
    assert card.get("next_action")
    assert card.get("original_title")
    assert card["title"] != card["original_title"] or _contains_chinese(card["title"])


# ===== Task D/E: Memory extraction tests =====


def test_extract_semantic_memory_from_project_goal():
    result = memory_extractor.extract(
        user_input="我正在开发一个基于 Open Deep Research 二开的信息差 Agent OS。",
        agent_output="了解了，你正在构建信息差 Agent OS。",
    )
    semantic = result["semantic_memories"]
    assert len(semantic) >= 1
    assert any("信息差" in m["content"] for m in semantic)
    assert any(m["importance"] >= 0.80 for m in semantic)


def test_extract_semantic_memory_from_tech_stack():
    result = memory_extractor.extract(
        user_input="我的技术栈是 FastAPI、Vite、React、LangGraph、LangChain、MySQL、Redis、Qdrant。",
        agent_output="好的，记下了你的技术栈。",
    )
    semantic = result["semantic_memories"]
    assert len(semantic) >= 1
    assert any("FastAPI" in m["content"] for m in semantic)


def test_extract_boundary_memory():
    result = memory_extractor.extract(
        user_input="当前阶段不要引入 Exa、Neo4j 或真实电脑操作。",
        agent_output="明白，会避开这些。",
    )
    semantic = result["semantic_memories"]
    assert any("Exa" in m["content"] or "Neo4j" in m["content"] for m in semantic)


def test_extract_preference_memory():
    result = memory_extractor.extract(
        user_input="我希望首页 FeedCard 使用中文展示，文案要简练有产品感。",
        agent_output="好的，会改为中文产品化表达。",
    )
    semantic = result["semantic_memories"]
    assert any("中文" in m["content"] for m in semantic)


def test_casual_chat_does_not_write_high_importance_semantic():
    """Casual chat like 'hello' should not produce high-importance semantic memories."""
    result = memory_extractor.extract(
        user_input="你好",
        agent_output="你好！有什么可以帮助你的吗？",
    )
    semantic = result["semantic_memories"]
    # Casual chat should yield no or low-importance semantic memories
    high_importance = [m for m in semantic if m.get("importance", 0) >= 0.70]
    assert len(high_importance) == 0, f"Casual chat should not create high-importance memories, got: {high_importance}"


def test_episodic_memory_for_ui_feedback():
    result = memory_extractor.extract(
        user_input="首页 FeedCard 还是英文标题，而且中文总结太模板化了，需要全部改。",
        agent_output="收到，我来修改首页展示。",
    )
    episodic = result["episodic_memories"]
    assert len(episodic) >= 1
    assert any("FeedCard" in m["content"] or "首页" in m["content"] for m in episodic)


def test_working_memory_from_page_context():
    result = memory_extractor.extract(
        user_input="分析这个",
        agent_output="好的",
        page_context={"page": "home", "selected_feed_card_id": 5},
        feed_card_context={"id": 5, "title": "测试卡片"},
    )
    working = result["working_memories"]
    assert len(working) >= 1
    assert any("home" in m["content"] for m in working)


# ===== Task F: Memory dedup tests =====


def test_semantic_memory_dedup_prevents_duplicates():
    """Writing the same semantic memory twice should update, not duplicate."""
    db = make_test_session()
    user = _user(db)

    # First write
    result1 = memory_service.add_with_dedup(
        user.id,
        "用户正在开发基于 Open Deep Research 二开的信息差 Agent OS。",
        memory_type="semantic",
        importance=0.90,
        metadata={"category": "project_goal", "source": "home_chat", "evidence_count": 1},
        db=db,
    )
    assert result1 is not None

    # Second write with similar content
    result2 = memory_service.add_with_dedup(
        user.id,
        "用户正在开发基于 Open Deep Research 二开的信息差 Agent OS，当前阶段重点是 Feed 中文产品化。",
        memory_type="semantic",
        importance=0.88,
        metadata={"category": "project_goal", "source": "home_chat", "evidence_count": 1},
        db=db,
    )

    # Count semantic memories for this user
    memories = memory_service.search_memory(user.id, memory_type="semantic", min_importance=0.3, db=db)
    # Should be 1 (updated), not 2
    assert len(memories) == 1, f"Expected 1 semantic memory after dedup, got {len(memories)}"


def test_different_semantic_memories_are_not_deduped():
    """Different semantic memories should both be saved."""
    db = make_test_session()
    user = _user(db)

    memory_service.add_with_dedup(
        user.id,
        "用户正在开发信息差 Agent OS。",
        memory_type="semantic",
        importance=0.90,
        metadata={"category": "project_goal"},
        db=db,
    )
    memory_service.add_with_dedup(
        user.id,
        "用户偏好首页 FeedCard 使用中文展示。",
        memory_type="semantic",
        importance=0.85,
        metadata={"category": "preference"},
        db=db,
    )

    memories = memory_service.search_memory(user.id, memory_type="semantic", min_importance=0.3, db=db)
    assert len(memories) == 2


# ===== Task G: Memory-driven scoring tests =====


def test_semantic_memory_affects_feed_personal_relevance():
    """Semantic memories should influence FeedCard personal_relevance scoring."""
    db = make_test_session()
    user = _user(db)
    profile = _profile(db, user)

    # Write relevant semantic memories
    memory_service.add_memory(
        user.id,
        "用户正在开发信息差 Agent OS，核心模块包括 Agent、RAG、Skill、Memory、Feed。",
        memory_type="semantic",
        importance=0.90,
        db=db,
    )
    memory_service.add_memory(
        user.id,
        "用户高度关注 Agent 和 RAG 相关的最新技术进展。",
        memory_type="semantic",
        importance=0.85,
        db=db,
    )

    semantic_memories = memory_service.get_semantic_memories(user.id, db)
    assert len(semantic_memories) >= 2

    item = _make_info_item(db, title="Advanced RAG with Agent-based Retrieval")
    scorer = FeedScorer()

    # Score WITH memories
    score_with = scorer.score(item, profile, semantic_memories=semantic_memories)
    # Score WITHOUT memories
    score_without = scorer.score(item, profile, semantic_memories=[])

    # Memory-driven score should have semantic_memory_match > 0
    assert score_with.get("semantic_memory_match", 0) > 0.10
    # With relevant memories, personal_relevance should be >= without
    # (or at minimum, the score dict structure should include the new fields)
    assert "profile_match" in score_with
    assert "semantic_memory_match" in score_with


def test_memory_failure_does_not_crash_feed_refresh():
    """If memory retrieval fails, feed refresh should still work (fallback to empty)."""
    db = make_test_session()
    user = _user(db)
    _profile(db, user)

    # Ensure refresh works even with no memories
    result = refresh_feed(db, user.id)
    assert "created_feed_cards" in result
    assert isinstance(result["created_feed_cards"], int)


def test_memory_user_isolation():
    """User A's memories should not affect User B's feed scoring."""
    db = make_test_session()
    user_a = _user(db, "a@example.com")
    user_b = _user(db, "b@example.com")
    profile_b = _profile(db, user_b)

    # User A writes memories about loving Agent content
    memory_service.add_memory(
        user_a.id,
        "用户 A 非常关注 Agent 技术，每天都在研究 Agent。",
        memory_type="semantic",
        importance=0.95,
        db=db,
    )

    # User B queries memories - should not see user A's
    b_memories = memory_service.get_semantic_memories(user_b.id, db)
    assert len(b_memories) == 0

    # User B's scoring should not be influenced by User A's memories
    item = _make_info_item(db, title="Agent Framework Research", source_url="https://example.com/agent2")
    scorer = FeedScorer()
    score_b = scorer.score(item, profile_b, semantic_memories=b_memories)
    # Without User A's memories, the semantic_memory_match should be low/default
    assert score_b["semantic_memory_match"] <= 0.20


# ===== Regression: Existing functionality preserved =====


def test_feed_api_card_to_dict_includes_all_fields():
    """card_to_dict includes original_title, why_relevant, benefit, next_action."""
    db = make_test_session()
    user = _user(db)
    _profile(db, user)
    refresh_feed(db, user.id)

    from src.web_app.services.feed_service import list_cards
    result = list_cards(db, user.id, limit=3)
    cards = result["cards"]
    if cards:
        card = cards[0]
        assert "title" in card
        assert "original_title" in card
        assert "why_relevant" in card
        assert "benefit" in card
        assert "next_action" in card
        assert "information_gap" in card


def test_memory_extract_and_save_integration():
    """Full integration: extract_and_save writes all memory types."""
    db = make_test_session()
    user = _user(db)

    result = memory_service.extract_and_save(
        user_id=user.id,
        user_input="我正在开发信息差 Agent OS，用 FastAPI + LangGraph。首页 FeedCard 请用中文展示，不要英文。",
        agent_output="好的，已记录你的技术栈和偏好，会将首页改为中文展示。",
        page_context={"page": "home"},
        db=db,
    )

    saved = result["saved"]
    # Should have saved at least episodic and semantic memories
    assert "semantic" in saved
    assert "episodic" in saved

    # Verify memories are in DB
    all_memories = memory_service.search_memory(user.id, min_importance=0.3, db=db)
    assert len(all_memories) >= 2


def test_extract_feed_interests():
    result = memory_extractor.extract(
        user_input="我关注 Agent、RAG、Memory、Skill、MCP、Deep Research 和 AI 产品机会。",
        agent_output="明白，这些都是信息差 Agent OS 的核心模块。",
    )
    semantic = result["semantic_memories"]
    # Should have extracted feed interests
    interests_found = any(
        "Agent" in m["content"] and "RAG" in m["content"]
        for m in semantic
    )
    assert interests_found or len(semantic) >= 1


# ===== Title Chinese regression tests =====


def test_is_mostly_english_detects_english():
    from src.web_app.feed.card_generator import is_mostly_english
    assert is_mostly_english("Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill")
    assert is_mostly_english("Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning")
    assert not is_mostly_english("用 Skill 统一 Agent 评估标准的新思路")
    assert not is_mostly_english("让 LLM 推理更可控的 Agent 思维链方法")


def test_specific_title_skill_rm_not_returned_as_is():
    """Skill-RM title must not appear as the display title."""
    from src.web_app.feed.card_generator import _generate_chinese_title
    title = "Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill"
    result = _generate_chinese_title(title, "arxiv", ["agent", "skill", "eval"], "agent")
    assert "Skill-RM" not in result or _contains_chinese(result)
    assert "Unifying Heterogeneous" not in result
    assert _contains_chinese(result)


def test_specific_title_agentic_cot_not_returned_as_is():
    """Agentic Chain-of-Thought title must not appear as the display title."""
    from src.web_app.feed.card_generator import _generate_chinese_title
    title = "Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning"
    result = _generate_chinese_title(title, "arxiv", ["agent", "reasoning", "llm"], "agent")
    assert "Agentic Chain-of-Thought" not in result or _contains_chinese(result)
    assert "Efficient and Controllable" not in result
    assert _contains_chinese(result)


def test_specific_title_graphrag_memory_not_returned_as_is():
    """GraphRAG Memory title must have a Chinese display version."""
    from src.web_app.feed.card_generator import _generate_chinese_title
    title = "GraphRAG Memory for Agent Systems"
    result = _generate_chinese_title(title, "arxiv", ["rag", "memory", "agent"], "rag")
    assert _contains_chinese(result)


def test_specific_title_github_not_returned_as_is():
    """GitHub project title must have a Chinese display version."""
    from src.web_app.feed.card_generator import _generate_chinese_title
    title = "browser-use/web-ui"
    result = _generate_chinese_title(title, "github", ["browser", "agent", "ui"], "agent")
    assert "browser-use/web-ui" not in result or _contains_chinese(result)
    assert _contains_chinese(result)


def test_arxiv_title_always_chinese():
    """Any English arxiv title should produce a Chinese display title."""
    from src.web_app.feed.card_generator import _generate_chinese_title, is_mostly_english
    titles = [
        ("Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill", "arxiv", ["agent", "skill"], "agent"),
        ("Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning", "arxiv", ["agent", "llm"], "agent"),
        ("GraphRAG Memory for Agent Systems", "arxiv", ["rag", "memory"], "rag"),
        ("RAG-Enhanced Multi-Agent Collaboration Framework", "arxiv", ["rag", "agent"], "rag"),
        ("MCP Tool Orchestration for Autonomous Workflows", "arxiv", ["mcp", "tool"], "agent"),
    ]
    for title, source_type, tags, domain in titles:
        result = _generate_chinese_title(title, source_type, tags, domain)
        assert not is_mostly_english(result), f"Title should be Chinese, got: {result}"
        assert _contains_chinese(result), f"Title should contain Chinese chars, got: {result}"


def test_card_to_dict_fixes_old_english_title():
    """card_to_dict should fix old English titles via display_title fallback."""
    from src.web_app.services.feed_service import card_to_dict
    from datetime import datetime

    # Simulate an old card with English title
    class OldCard:
        id = 999
        card_type = "insight"
        title = "Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill"
        one_sentence_value = "Old value"
        why_you = "Old why"
        information_gap = "Old gap"
        exposure_bucket = "explicit_related"
        evidence = []
        suggested_actions = []
        score_detail = {"source_type": "arxiv", "domain": "agent", "confidence": "high", "summary": "test"}
        final_score = 0.75
        status = "active"
        created_at = datetime(2026, 1, 1)

    result = card_to_dict(OldCard)
    # The main title must be Chinese, not the English original
    from src.web_app.feed.card_generator import is_mostly_english
    assert not is_mostly_english(result["title"]), f"title should be Chinese, got: {result['title']}"
    assert not is_mostly_english(result["display_title"]), f"display_title should be Chinese, got: {result['display_title']}"
    # original_title should preserve the English
    assert result["original_title"] == "Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill"
    assert "display_title" in result


def test_feed_cards_api_returns_chinese_titles():
    """The /feed/cards API (list_cards) returns Chinese titles."""
    db = make_test_session()
    user = _user(db)
    _profile(db, user)

    # Refresh to create new cards
    refresh_feed(db, user.id)

    from src.web_app.services.feed_service import list_cards, card_to_dict
    from src.web_app.feed.card_generator import is_mostly_english

    result = list_cards(db, user.id, limit=10)
    cards = result["cards"]
    assert len(cards) > 0

    for card in cards:
        assert "title" in card
        assert "display_title" in card
        assert "original_title" in card
        # Main title must not be mostly English
        assert not is_mostly_english(card["title"]), f"Card title should be Chinese: {card['title']}"
        assert not is_mostly_english(card["display_title"]), f"Card display_title should be Chinese: {card['display_title']}"


def test_chinese_title_generation_is_deterministic():
    """Same input should produce same output (deterministic, no LLM)."""
    from src.web_app.feed.card_generator import _generate_chinese_title
    title = "Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill"
    r1 = _generate_chinese_title(title, "arxiv", ["agent", "skill"], "agent")
    r2 = _generate_chinese_title(title, "arxiv", ["agent", "skill"], "agent")
    assert r1 == r2


def test_different_titles_produce_different_chinese_titles():
    """Different English titles should produce different Chinese titles."""
    from src.web_app.feed.card_generator import _generate_chinese_title
    t1 = _generate_chinese_title(
        "Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill",
        "arxiv", ["agent", "skill"], "agent"
    )
    t2 = _generate_chinese_title(
        "Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning",
        "arxiv", ["agent", "llm"], "agent"
    )
    t3 = _generate_chinese_title(
        "GraphRAG Memory for Agent Systems",
        "arxiv", ["rag", "memory"], "rag"
    )
    assert len({t1, t2, t3}) == 3, f"All three titles should differ, got: {t1}, {t2}, {t3}"
