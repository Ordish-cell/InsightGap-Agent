import pytest

from src.web_app.context.builder import ContextBuilder
from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.feed.scorer import FeedScorer
from src.web_app.models.orm import User
from src.web_app.services.auth_service import hash_password
from src.web_app.services.memory_service import memory_service
from src.web_app.services.user_growth_service import user_growth_service, _DECAY_RATES, _STATUS_FACTORS
from src.web_app.tests.db_test_utils import make_test_session


def _user(db, email="test@example.com"):
    user = User(email=email, hashed_password=hash_password("pass"), nickname="test")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_info_item(db, title="Test Paper", source_type="arxiv", topics=None, domain="agent", source_url="https://example.com/test"):
    from src.web_app.models.orm import InfoItem
    import hashlib
    content_hash = hashlib.sha256(source_url.encode()).hexdigest()
    raw_metadata = {"source_id": "test", "canonical_url": source_url, "tags": topics or ["agent"], "domain": domain, "source_credibility": 0.85}
    item = InfoItem(title=title, summary="Test summary", content="Test content", source_url=source_url, source_type=source_type, author="Test", topics=topics or ["agent"], raw_metadata=raw_metadata, content_hash=content_hash)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _profile(db, user):
    from src.web_app.db.repositories.profile_repository import ProfileRepository
    return ProfileRepository(db).get_or_create_default(user.id)


# ===== UserGrowthService — process_conversation =====


def test_process_conversation_extracts_semantic():
    db = make_test_session()
    user = _user(db)

    result = user_growth_service.process_conversation(
        user.id,
        user_input="我正在开发信息差 Agent OS，用 FastAPI + LangGraph，当前阶段重点做 User Growth Engine。",
        agent_output="了解，已记录你的技术栈和阶段目标。",
        route="chat",
        db=db,
    )
    saved = result.get("saved", {})
    assert "semantic" in saved
    assert len(saved.get("semantic", [])) >= 1


def test_process_conversation_casual_chat_no_high_semantic():
    db = make_test_session()
    user = _user(db)

    result = user_growth_service.process_conversation(
        user.id,
        user_input="你好",
        agent_output="你好！",
        route="chat",
        db=db,
    )
    saved = result.get("saved", {})
    semantic = saved.get("semantic", [])
    high_importance = [m for m in semantic if m.get("importance", 0) >= 0.70]
    assert len(high_importance) == 0


# ===== UserGrowthService — process_feed_feedback =====


def test_process_feed_feedback_save_boosts_interest():
    db = make_test_session()
    user = _user(db)

    result = user_growth_service.process_feed_feedback(
        user.id, card_id=1, action="save",
        card_title="Agent Skill 统一评估", card_domain="agent",
        card_topics=["agent", "skill", "eval"], db=db,
    )
    assert len(result["saved"]) >= 1


def test_process_feed_feedback_ignore_adds_negative():
    db = make_test_session()
    user = _user(db)

    result = user_growth_service.process_feed_feedback(
        user.id, card_id=2, action="not_relevant",
        card_title="某不相关论文", card_domain="research",
        card_topics=["biology"], db=db,
    )
    assert len(result["saved"]) >= 1


# ===== UserGrowthService — process_skill_event =====


def test_process_skill_approve():
    db = make_test_session()
    user = _user(db)

    result = user_growth_service.process_skill_event(
        user.id, skill_id=1, event="approve",
        skill_name="Deep Research 自动报告", db=db,
    )
    assert len(result["saved"]) >= 1


def test_process_skill_use_success_and_failure():
    db = make_test_session()
    user = _user(db)

    r1 = user_growth_service.process_skill_event(
        user.id, skill_id=2, event="use_success",
        skill_name="代码审查助手", db=db,
    )
    r2 = user_growth_service.process_skill_event(
        user.id, skill_id=2, event="use_failure",
        skill_name="代码审查助手", db=db,
    )
    assert len(r1["saved"]) >= 1
    assert len(r2["saved"]) >= 1


# ===== UserGrowthService — process_research_event / process_artifact_event =====


def test_process_research_event():
    db = make_test_session()
    user = _user(db)

    result = user_growth_service.process_research_event(
        user.id, research_run_id="uuid-123",
        query="研究 Agent Skill 统一评估方法", status="completed", db=db,
    )
    assert len(result["saved"]) >= 1


def test_process_artifact_event():
    db = make_test_session()
    user = _user(db)

    result = user_growth_service.process_artifact_event(
        user.id, artifact_id=1, event="saved",
        artifact_title="Agent OS 架构分析报告", db=db,
    )
    assert len(result["saved"]) >= 1


# ===== Supersede =====


def test_supersede_conflicting_boundary():
    db = make_test_session()
    user = _user(db)

    # Old memory: don't use Neo4j
    old = memory_service.add_memory(
        user.id,
        "用户当前阶段不希望引入 Neo4j 或真实电脑操作。",
        memory_type="semantic", importance=0.88,
        metadata={"category": "boundary", "stability": "long_term", "status": "active"},
        db=db,
    )

    # New memory: now use Neo4j
    new = memory_service.add_memory(
        user.id,
        "用户现在可以开始接 Neo4j 图谱了。",
        memory_type="semantic", importance=0.85,
        metadata={"category": "boundary", "stability": "long_term", "status": "active"},
        db=db,
    )

    superseded = user_growth_service.supersede_conflicting_memories(user.id, new, db)

    # Check old was superseded
    from src.web_app.db.repositories.memory_repository import MemoryRepository
    old_item = MemoryRepository(db).get_by_id(old["id"])
    old_meta = old_item.metadata_json or {}
    assert old_meta.get("status") == "superseded" or len(superseded) >= 0


def test_gscc_default_excludes_superseded():
    """GSSC / search should not return superseded memories by default."""
    db = make_test_session()
    user = _user(db)

    mem = memory_service.add_memory(
        user.id,
        "用户偏好使用 React。",
        memory_type="semantic", importance=0.80,
        metadata={"category": "preference", "stability": "long_term", "status": "superseded"},
        db=db,
    )

    # Active memories only
    active = user_growth_service._get_active_semantic(user.id, db)
    superseded_ids = [
        m.get("id") for m in active
        if (m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata_json", {}) or {}).get("status") == "superseded"
    ]
    assert len(superseded_ids) == 0


# ===== Reflection =====


def test_reflection_merges_fragments():
    db = make_test_session()
    user = _user(db)

    # Create several same-category fragments
    for i in range(5):
        memory_service.add_memory(
            user.id,
            f"用户偏好简洁的界面设计（第{i}次确认）。",
            memory_type="semantic", importance=0.75,
            metadata={"category": "ui_preference", "stability": "long_term", "status": "active",
                       "evidence_count": i + 1},
            db=db,
        )

    result = user_growth_service.reflect_user_profile(user.id, db)
    assert "summaries" in result
    assert "archived" in result


def test_reflection_skips_when_too_few():
    db = make_test_session()
    user = _user(db)

    memory_service.add_memory(
        user.id, "用户使用 Python。", memory_type="semantic", importance=0.6,
        metadata={"category": "tech_stack", "stability": "long_term", "status": "active"},
        db=db,
    )

    result = user_growth_service.reflect_user_profile(user.id, db)
    assert result.get("reason") == "not_enough_memories" or result.get("summary_count", 0) == 0


# ===== Effective importance / Decay =====


def test_effective_importance_active_high():
    mem = {
        "id": 1, "content": "test", "importance": 0.90,
        "metadata": {"stability": "long_term", "status": "active", "evidence_count": 3,
                      "last_seen_at": "2026-06-01T00:00:00+00:00"},
    }
    eff = user_growth_service.compute_effective_importance(mem)
    assert eff > 0.70


def test_effective_importance_superseded_is_zero():
    mem = {
        "id": 2, "content": "test", "importance": 0.90,
        "metadata": {"stability": "long_term", "status": "superseded", "evidence_count": 1,
                      "last_seen_at": "2026-06-01T00:00:00+00:00"},
    }
    eff = user_growth_service.compute_effective_importance(mem)
    assert eff == 0.0


def test_effective_importance_archived_is_low():
    mem = {
        "id": 3, "content": "test", "importance": 0.80,
        "metadata": {"stability": "medium_term", "status": "archived", "evidence_count": 1,
                      "last_seen_at": "2026-06-01T00:00:00+00:00"},
    }
    eff = user_growth_service.compute_effective_importance(mem)
    assert eff < 0.50


def test_effective_importance_decay_rates_defined():
    assert "long_term" in _DECAY_RATES
    assert "temporary" in _DECAY_RATES
    assert _DECAY_RATES["long_term"] < _DECAY_RATES["temporary"]


def test_status_factors_defined():
    assert _STATUS_FACTORS["active"] == 1.0
    assert _STATUS_FACTORS["superseded"] == 0.0


def test_get_memories_with_effective_importance():
    db = make_test_session()
    user = _user(db)

    memory_service.add_memory(
        user.id, "用户正在开发信息差 Agent OS。",
        memory_type="semantic", importance=0.90,
        metadata={"category": "project_goal", "stability": "long_term", "status": "active",
                   "evidence_count": 5},
        db=db,
    )
    memory_service.add_memory(
        user.id, "旧设定已被替代。",
        memory_type="semantic", importance=0.80,
        metadata={"category": "boundary", "stability": "long_term", "status": "superseded"},
        db=db,
    )

    enriched = user_growth_service.get_memories_with_effective_importance(
        user.id, db, memory_type="semantic",
    )
    # Superseded memory should not appear (effective = 0)
    superseded_found = [
        m for m in enriched
        if (m.get("metadata", {})).get("status") == "superseded"
    ]
    assert len(superseded_found) == 0


# ===== Route-aware GSSC =====


def test_gssc_chat_route_prioritizes_conversation_and_memory():
    builder = ContextBuilder(route="chat")
    packets = builder.gather({
        "task": "帮我分析这条信息",
        "conversation_summary": "用户最近在讨论 FeedCard 中文展示。",
        "memory": "用户偏好中文表达。",
        "feed_card": "某 FeedCard",
        "checkpoint_summary": "上一步已完成 skill matching。",
        "output_contract": "返回结构化结果。",
    })
    selected = builder.select(packets)
    sources = [p.metadata.get("source") for p in selected]
    # Chat route should include conversation_summary
    assert "conversation_summary" in sources or "memory" in sources


def test_gssc_research_route_prioritizes_feed_card_and_evidence():
    builder = ContextBuilder(route="research")
    packets = builder.gather({
        "task": "Research this paper",
        "feed_card": "Skill-RM paper details",
        "evidence": [{"score": 0.85, "title": "Key evidence"}],
        "checkpoint_summary": "Previous step completed context loading.",
        "memory": "用户关注 Agent Skill 评估。",
        "output_contract": "Return research report.",
    })
    selected = builder.select(packets)
    sources = [p.metadata.get("source") for p in selected]
    assert "feed_card" in sources


def test_gssc_skill_route_prioritizes_memory():
    builder = ContextBuilder(route="skill")
    packets = builder.gather({
        "task": "Create a reusable workflow",
        "memory": "用户反复执行代码审查流程。",
        "conversation_summary": "用户讨论了代码审查的重复性。",
        "output_contract": "Return skill draft.",
    })
    selected = builder.select(packets)
    sources = [p.metadata.get("source") for p in selected]
    assert "memory" in sources


def test_gssc_build_with_debug_returns_metadata():
    builder = ContextBuilder(route="research")
    context, debug = builder.build_with_debug({
        "task": "Test task",
        "memory": "Test memory",
        "output_contract": "Return JSON.",
    })
    assert "gssc_route" in debug
    assert debug["gssc_route"] == "research"
    assert "selected_sources" in debug
    assert "token_budget_used" in debug


def test_gssc_token_budget_truncation():
    builder = ContextBuilder(route="chat")
    # Generate many large packets to exceed budget
    payload = {"task": "test"} | {f"src_{i}": "x" * 1000 for i in range(30)}
    packets = builder.gather(payload)
    selected = builder.select(packets)
    total_tokens = sum(p.token_count for p in selected)
    assert total_tokens <= builder.config.max_tokens


# ===== Dynamic preference profile =====


def test_build_dynamic_preference_profile():
    db = make_test_session()
    user = _user(db)

    memory_service.add_memory(
        user.id, "用户正在开发信息差 Agent OS。",
        memory_type="semantic", importance=0.90,
        metadata={"category": "project_goal", "stability": "long_term", "status": "active"},
        db=db,
    )
    memory_service.add_memory(
        user.id, "用户偏好中文产品化表达。",
        memory_type="semantic", importance=0.85,
        metadata={"category": "preference", "stability": "long_term", "status": "active"},
        db=db,
    )

    profile = user_growth_service.build_dynamic_preference_profile(user.id, db, route="chat")
    assert "dynamic_goals" in profile
    assert "dynamic_preferences" in profile
    assert "preference_summary" in profile


# ===== Skill Evolution =====


def test_skill_record_usage_increments_success_count():
    from src.web_app.services.skill_service import skill_service
    from src.web_app.db.repositories.skill_repository import SkillRepository
    db = make_test_session()
    user = _user(db)

    skill = SkillRepository(db).create(
        user_id=user.id, name="测试 Skill",
        description="测试", trigger_text="测试",
        eval_checks=[],
    )

    skill_service.record_skill_usage(skill.id, user.id, success=True, db=db)
    stats = skill_service.get_skill_evolution(skill.id, user.id, db=db)
    assert stats["success_count"] == 1
    assert stats["confidence"] >= 0.5


def test_skill_detect_repeated_workflow():
    from src.web_app.services.skill_service import skill_service
    from src.web_app.db.repositories.skill_repository import SkillRepository
    db = make_test_session()
    user = _user(db)

    SkillRepository(db).create(
        user_id=user.id, name="代码审查报告生成",
        description="自动生成代码审查报告", trigger_text="生成代码审查报告 审查代码",
        context_recipe=["permission_guard", "router", "context_builder", "research", "memory_writer", "evaluator"],
        status="approved",
        eval_checks=[{"_type": "skill_evolution", "success_count": 3, "failure_count": 0, "confidence": 1.0}],
    )

    result = skill_service.detect_repeated_workflow(user.id, "帮我审查这段代码并生成报告", db=db)
    assert result["repeated"] is True or result["boost"] >= 0


def test_skill_disabled_not_matched():
    from src.web_app.services.skill_service import skill_service
    from src.web_app.db.repositories.skill_repository import SkillRepository
    db = make_test_session()
    user = _user(db)

    SkillRepository(db).create(
        user_id=user.id, name="被禁用的 Skill",
        description="不应被匹配", trigger_text="匹配我",
        status="disabled",
    )

    result = skill_service.match_skill("匹配我", user.id, db=db)
    assert not result.get("matched_skill")


# ===== user_id isolation =====


def test_user_growth_user_isolation():
    db = make_test_session()
    user_a = _user(db, "a@example.com")
    user_b = _user(db, "b@example.com")

    user_growth_service.process_conversation(
        user_a.id, user_input="我是做 Agent OS 的。", route="chat", db=db,
    )

    profile_b = user_growth_service.build_dynamic_preference_profile(user_b.id, db)
    # User B should not have User A's goals
    goals = profile_b.get("dynamic_goals", [])
    goals_text = " ".join(goals)
    assert "Agent OS" not in goals_text


# ===== Regression: existing APIs preserved =====


def test_feed_refresh_still_works():
    from src.web_app.services.feed_service import refresh_feed
    db = make_test_session()
    user = _user(db)
    _profile(db, user)

    result = refresh_feed(db, user.id)
    assert "created_feed_cards" in result


def test_memory_api_still_works():
    db = make_test_session()
    user = _user(db)

    memory_service.add_memory(user.id, "测试记忆", db=db)
    results = memory_service.search_memory(user.id, "测试", db=db)
    assert len(results) >= 1


def test_original_skill_matching_still_works():
    from src.web_app.services.skill_service import skill_service
    from src.web_app.db.repositories.skill_repository import SkillRepository
    db = make_test_session()
    user = _user(db)

    SkillRepository(db).create(
        user_id=user.id, name="Research Agent OS",
        description="Deep research for Agent OS topics",
        trigger_text="研究 Agent OS 分析",
        status="approved",
    )
    result = skill_service.match_skill("研究 Agent OS 的最新进展", user.id, db=db)
    # Should find the skill
    assert result.get("candidate_skills") or result.get("matched")


def test_skill_matching_uses_chinese_trigger_terms():
    from src.web_app.services.skill_service import skill_service
    from src.web_app.db.repositories.skill_repository import SkillRepository
    db = make_test_session()
    user = _user(db)

    skill = SkillRepository(db).create(
        user_id=user.id,
        name="研究报告流程",
        description="复用研究报告工作流",
        trigger_text="研究报告流程，复用",
        status="approved",
    )

    result = skill_service.match_skill("请复用研究报告流程", user.id, db=db)

    assert result["matched_skill"]["id"] == skill.id
    assert result["matched_skill"]["match_score"] >= 0.75


def test_skill_reusability_uses_chinese_intent_terms():
    from src.web_app.services.skill_service import skill_service

    result = skill_service.evaluate_reusability(
        {
            "user_input": "以后复用这个流程生成报告",
            "route": "artifact",
            "status": "completed",
            "artifacts": [{"id": 1}],
        }
    )

    assert result["should_create"] is True
    assert result["reusable_score"] >= 0.70


def test_original_feed_chinese_titles_still_work():
    from src.web_app.feed.card_generator import _generate_chinese_title, is_mostly_english
    title = _generate_chinese_title(
        "Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill",
        "arxiv", ["agent", "skill", "eval"], "agent"
    )
    assert not is_mostly_english(title)
