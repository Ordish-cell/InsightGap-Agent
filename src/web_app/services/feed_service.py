import asyncio
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.core.config import settings
from src.web_app.db.repositories.agent_repository import AgentRunRepository
from src.web_app.db.repositories.feed_repository import FeedFeedbackRepository, FeedRepository
from src.web_app.db.repositories.info_repository import InfoItemRepository
from src.web_app.db.repositories.profile_repository import ProfileRepository
from src.web_app.feed.card_generator import generate_display_title, generate_feed_card, is_mostly_english
from src.web_app.feed.dedup import deduplicate_items
from src.web_app.feed.mixer import mix_cards
from src.web_app.feed.normalizer import normalize_raw_item
from src.web_app.feed.scorer import FeedScorer
from src.web_app.feed.sources.manager import SearchSourceManager
from src.web_app.services.memory_service import memory_service

FEEDBACK_ACTIONS = {"save", "ignore", "useful", "not_relevant", "open", "deep_research", "generate_report", "create_skill_draft"}


def refresh_feed(db: Session, user_id: int) -> dict:
    profile = _profile_with_defaults(ProfileRepository(db).get_or_create_default(user_id))
    raw_items, source_stats = _run_async(SearchSourceManager().fetch_all())
    normalized = [item for item in (normalize_raw_item(raw) for raw in raw_items) if item]
    unique_items, skipped_duplicates = deduplicate_items(normalized)
    info_repo = InfoItemRepository(db)
    feed_repo = FeedRepository(db)
    feedback_stats = FeedFeedbackRepository(db).get_user_feedback_stats(user_id)
    semantic_memories = _safe_get_semantic_memories(user_id, db)
    scorer = FeedScorer()
    created_info = updated_info = 0
    candidate_cards: list[dict[str, Any]] = []

    for item in unique_items[: settings.feed_refresh_max_items]:
        info_item, created = info_repo.upsert_by_hash(
            title=item.title,
            summary=item.summary,
            content=item.content,
            source_url=item.source_url,
            source_type=item.source_type,
            author=item.author,
            published_at=item.published_at,
            language="zh",
            entities=[],
            topics=item.topics,
            raw_metadata=item.raw_metadata,
            content_hash=item.content_hash,
        )
        created_info += int(created)
        updated_info += int(not created)
        score = scorer.score(info_item, profile, feedback_stats, semantic_memories)
        if score.get("filtered"):
            continue
        card = generate_feed_card(info_item, score, profile)
        card["info_item_id"] = info_item.id
        candidate_cards.append(card)

    mixed = mix_cards(candidate_cards, profile.feed_ratio_config, settings.feed_card_limit_default)
    rows = []
    for card in mixed:
        score_detail = dict(card["score"])
        score_detail.update({
            "source_type": card["source_type"],
            "domain": card["domain"],
            "confidence": card["confidence"],
            "summary": card["summary"],
            "original_title": card.get("original_title", card["title"]),
            "why_relevant": card.get("why_relevant", card.get("why_you", "")),
            "benefit": card.get("benefit", ""),
            "next_action": card.get("next_action", ""),
        })
        rows.append({
            "user_id": user_id,
            "info_item_id": card["info_item_id"],
            "card_type": card["card_type"],
            "title": card["title"],
            "one_sentence_value": card["one_sentence_value"],
            "why_you": card["why_you"],
            "information_gap": card["information_gap"],
            "evidence": card["evidence"],
            "suggested_actions": card["suggested_actions"],
            "score_detail": score_detail,
            "final_score": card["final_score"],
            "exposure_bucket": card["relation_type"],
            "status": "active",
        })
    created_cards = feed_repo.bulk_create(rows) if rows else []
    return {
        "created_info_items": created_info,
        "updated_info_items": updated_info,
        "created_feed_cards": len(created_cards),
        "skipped_duplicates": skipped_duplicates,
        "source_stats": source_stats,
    }


def list_cards(db: Session, user_id: int, status: str | None = None, exposure_bucket: str | None = None, limit: int = 20, offset: int = 0, source_type: str | None = None, domain: str | None = None) -> dict:
    profile = _profile_with_defaults(ProfileRepository(db).get_or_create_default(user_id))
    cards = FeedRepository(db).list_by_user(user_id, status, exposure_bucket, limit, offset, source_type, domain)
    return {"cards": [card_to_dict(card) for card in cards], "next_cursor": offset + len(cards) if len(cards) == limit else None, "mix": profile.feed_ratio_config}


def list_home_cards(db: Session, user_id: int) -> dict:
    count = settings.feed_home_card_count
    cards = FeedRepository(db).list_by_user(user_id, status=None, limit=count)
    refresh_result: dict[str, Any] | None = None
    errors: list[str] = []
    if len(cards) < count and settings.feed_refresh_on_home_empty:
        try:
            refresh_result = refresh_feed(db, user_id)
            cards = FeedRepository(db).list_by_user(user_id, status=None, limit=count)
        except Exception as exc:
            errors.append(str(exc)[:300])
    if len(cards) < count and settings.feed_require_real_cards:
        return {
            "cards": [card_to_dict(card) for card in cards],
            "required_count": count,
            "is_complete": False,
            "error": "NOT_ENOUGH_REAL_FEED_CARDS",
            "message": f"Only {len(cards)} real FeedCards are available. Mock cards are disabled.",
            "refresh_result": refresh_result,
            "errors": errors,
        }
    return {
        "cards": [card_to_dict(card) for card in cards[:count]],
        "required_count": count,
        "is_complete": len(cards) >= count,
        "refresh_result": refresh_result,
        "errors": errors,
    }


def get_card_detail(db: Session, user_id: int, card_id: int) -> dict:
    card = FeedRepository(db).get_by_user(user_id, card_id)
    if not card:
        raise ValueError("Feed card not found")
    return card_to_dict(card)


def feedback(db: Session, user_id: int, card_id: int, action: str, metadata: dict | None = None) -> dict:
    if action not in FEEDBACK_ACTIONS:
        raise ValueError("Invalid feedback action")
    repo = FeedRepository(db)
    card = repo.get_by_user(user_id, card_id)
    if not card:
        raise ValueError("Feed card not found")
    FeedFeedbackRepository(db).create_feedback(user_id, card_id, action, metadata)
    if action == "save":
        repo.update(card, status="saved")
    elif action == "ignore":
        repo.update(card, status="ignored")
    elif action == "deep_research":
        repo.update(card, status="researched")
    return {"success": True, "action": action, "card_id": card_id}


def research_from_card(db: Session, user_id: int, card_id: int) -> dict:
    card = FeedRepository(db).get_by_user(user_id, card_id)
    if not card:
        raise ValueError("Feed card not found")
    run = AgentRunRepository(db).create(user_id=user_id, run_type="deep_research", mode="plan_and_solve", status="created", user_input=f"Research feed card {card_id}: {card.title}", graph_state={"card_id": card_id})
    return {"status": "not_implemented", "message": "Deep Research will be implemented in stage 5", "card_id": card_id, "run_id": run.id}


def source_health() -> dict:
    return SearchSourceManager().health()


def stats(db: Session, user_id: int) -> dict:
    repo = FeedRepository(db)
    cards = repo.list_by_user(user_id, limit=1000, include_hidden=True)
    saved = sum(1 for card in cards if card.status == "saved")
    hidden = sum(1 for card in cards if card.status == "ignored")
    avg_score = round(sum(card.final_score for card in cards) / len(cards), 4) if cards else 0
    return {"cards_count": len(cards), "saved_count": saved, "hidden_count": hidden, "average_final_score": avg_score, **repo.stats_for_user(user_id)}


def card_to_dict(card) -> dict:
    detail = card.score_detail or {}
    score_fields = {"personal_relevance", "novelty", "cross_domain_distance", "opportunity_value", "source_credibility", "actionability", "final", "profile_match", "semantic_memory_match"}

    db_title = card.title or ""
    original_title = detail.get("original_title", "")
    source_type = detail.get("source_type", "")
    domain = detail.get("domain", "ai")

    # If no original_title stored, the DB title IS the original (pre-Phase11B cards)
    if not original_title:
        original_title = db_title

    # Determine display title: use DB title if already Chinese, else generate
    if is_mostly_english(db_title):
        display_title = generate_display_title(db_title, source_type, [], domain)
    else:
        display_title = db_title

    return {
        "id": str(card.id),
        "card_type": card.card_type,
        "title": display_title,
        "display_title": display_title,
        "original_title": original_title,
        "one_sentence_value": card.one_sentence_value,
        "why_you": card.why_you,
        "why_relevant": detail.get("why_relevant", card.why_you),
        "benefit": detail.get("benefit", ""),
        "information_gap": card.information_gap,
        "next_action": detail.get("next_action", ""),
        "summary": detail.get("summary", ""),
        "source_type": detail.get("source_type", ""),
        "source_name": detail.get("source_name", detail.get("source_type", "")),
        "source_url": _first_source_url(card),
        "published_at": detail.get("published_at"),
        "fetched_at": card.created_at.isoformat() if card.created_at else None,
        "domain": detail.get("domain", ""),
        "relation_type": card.exposure_bucket,
        "exposure_bucket": card.exposure_bucket,
        "evidence": card.evidence or [],
        "suggested_actions": card.suggested_actions or [],
        "score": {key: value for key, value in detail.items() if key in score_fields},
        "score_detail": {key: value for key, value in detail.items() if key not in score_fields and key not in ("summary",)},
        "final_score": card.final_score,
        "confidence": detail.get("confidence", "medium"),
        "status": card.status,
        "created_at": card.created_at.isoformat() if card.created_at else None,
    }


def update_card_status(db: Session, user_id: int, card_id: int, status: str) -> dict:
    action = {"viewed": "open", "saved": "save", "ignored": "ignore", "researched": "deep_research"}.get(status, status)
    feedback(db, user_id, card_id, action)
    return get_card_detail(db, user_id, card_id)


def _profile_with_defaults(profile):
    if not profile.explicit_interests:
        profile.explicit_interests = ["LangGraph", "LangChain", "RAG", "MCP", "Agent"]
    if not profile.goals:
        profile.goals = ["开发个人信息差 Agent OS", "二开 Open Deep Research"]
    if not profile.adjacent_domains:
        profile.adjacent_domains = ["Agent UI", "AI 浏览器", "知识库", "自动化工作流"]
    if not profile.far_domains:
        profile.far_domains = ["创业机会", "行业情报", "投资研究", "教育产品"]
    return profile


def _safe_get_semantic_memories(user_id: int, db: Session) -> list[dict[str, Any]]:
    try:
        return memory_service.get_semantic_memories(user_id, db)
    except Exception:
        return []


def _first_source_url(card) -> str:
    detail = card.score_detail or {}
    if detail.get("source_url"):
        return str(detail["source_url"])
    for item in card.evidence or []:
        if isinstance(item, dict) and (item.get("source_url") or item.get("url")):
            return str(item.get("source_url") or item.get("url"))
    return ""


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
