import asyncio
import logging
import re
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import text
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
from src.web_app.feed.sources.manager import BUCKET_ORDER, SearchSourceManager
from src.web_app.services.memory_service import memory_service

logger = logging.getLogger(__name__)

FEEDBACK_ACTIONS = {"save", "ignore", "useful", "not_relevant", "open", "deep_research", "generate_report", "create_skill_draft"}

_BATCH_TARGETS = {
    "explicit_related": 2,
    "adjacent_domain": 2,
    "far_domain": 1,
}
_MAX_RETRY_ROUNDS = 2


def _canonicalize_url(url: str | None) -> str:
    """Canonicalize URL for dedup: lowercase, strip trailing /, strip utm/fragment."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return ""
    query_parts = parts.query.split("&")
    clean_query = "&".join(p for p in query_parts if p and not p.startswith("utm_"))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), clean_query, ""))


def _normalize_title_key(title: str) -> str:
    """Normalize title for fuzzy dedup: lowercase, strip punctuation, merge whitespace, first 120 chars."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:120]


def _deduplicate_candidate_cards(cards: list[dict]) -> list[dict]:
    """Dedup candidate cards by info_item_id, source_url, then normalized title."""
    if not cards:
        return cards
    before = len(cards)
    seen_item_ids: set[int] = set()
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    result: list[dict] = []

    for card in cards:
        item_id = card.get("info_item_id")
        if item_id:
            if item_id in seen_item_ids:
                logger.info("feed dedup removed duplicate info_item_id=%s title=%s", item_id, str(card.get("original_title", card.get("title", "")))[:80])
                continue
            seen_item_ids.add(item_id)

        source_url = _canonicalize_url(str(card.get("source_url", "")) or str((card.get("evidence") or [{}])[0].get("url", "")))
        if source_url:
            if source_url in seen_urls:
                logger.info("feed dedup removed duplicate source_url=%s title=%s", source_url[:80], str(card.get("original_title", ""))[:80])
                continue
            seen_urls.add(source_url)

        title_key = _normalize_title_key(str(card.get("original_title", card.get("title", ""))))
        if title_key:
            if title_key in seen_titles:
                logger.info("feed dedup removed duplicate title=%s", str(card.get("original_title", ""))[:80])
                continue
            seen_titles.add(title_key)

        result.append(card)

    removed = before - len(result)
    if removed:
        logger.info("feed dedup batch: before=%d after=%d removed=%d", before, len(result), removed)
    return result


def _acquire_refresh_lock(db: Session, user_id: int) -> bool:
    """Try to acquire a session-level PostgreSQL advisory lock for feed refresh. Returns True if acquired."""
    lock_id = hash(f"feed_refresh:{user_id}") % (2 ** 31)
    result = db.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": lock_id}).scalar()
    return bool(result)


def _release_refresh_lock(db: Session, user_id: int) -> None:
    """Release the session-level advisory lock."""
    lock_id = hash(f"feed_refresh:{user_id}") % (2 ** 31)
    db.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": lock_id})


def maybe_refresh_for_user(db: Session, user_id: int, force: bool = False) -> dict[str, Any]:
    """Trigger feed refresh if not already done today, or force. Returns refresh result."""
    profile_repo = ProfileRepository(db)
    profile = profile_repo.get_by_user(user_id)
    if not profile:
        return {"refreshed": False, "reason": "no_profile", "batch_id": None, "created_count": 0, "bucket_counts": {}, "missing_buckets": []}

    today = date.today()
    if not force and profile.last_feed_refreshed_at and profile.last_feed_refreshed_at.date() == today:
        return {"refreshed": False, "reason": "already_refreshed_today", "batch_id": None, "created_count": 0, "bucket_counts": {}, "missing_buckets": [], "last_refreshed_at": profile.last_feed_refreshed_at.isoformat()}

    locked = _acquire_refresh_lock(db, user_id)
    if not locked:
        return {"refreshed": False, "reason": "refresh_in_progress", "batch_id": None, "created_count": 0, "bucket_counts": {}, "missing_buckets": []}

    batch_id = uuid.uuid4().hex[:12]
    try:
        result = refresh_feed(db, user_id, limit=settings.feed_refresh_total_limit, batch_id=batch_id)
        is_complete = result.get("is_complete", False)
        if is_complete:
            profile.last_feed_refreshed_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        return {
            "refreshed": True,
            "reason": "force" if force else "daily",
            "batch_id": batch_id,
            "created_count": result.get("created_feed_cards", 0),
            "bucket_counts": result.get("bucket_counts", {}),
            "missing_buckets": result.get("missing_buckets", []),
            "is_complete": is_complete,
            "source_stats": result.get("source_stats", {}),
        }
    except Exception as exc:
        db.rollback()
        logger.exception("feed refresh failed for user %d", user_id)
        return {"refreshed": False, "reason": "refresh_failed", "error": str(exc)[:300], "batch_id": None, "created_count": 0, "bucket_counts": {}, "missing_buckets": [], "is_complete": False}
    finally:
        if locked:
            try:
                _release_refresh_lock(db, user_id)
            except Exception:
                logger.warning("release feed refresh lock failed", exc_info=True)


def _process_items_into_cards(
    unique_items: list, info_repo: InfoItemRepository, profile: Any,
    feedback_stats: dict, semantic_memories: list, scorer: FeedScorer, max_items: int,
    db: Session,
) -> tuple[list[dict], int, int]:
    """Process items into candidate cards. Returns (cards, created_info, updated_info)."""
    candidate_cards: list[dict[str, Any]] = []
    created_info = updated_info = 0

    for item in unique_items[:max_items * 3]:
        logger.debug(
            "feed item lengths: title=%d summary=%d content=%d source_url=%d author=%d hash=%s",
            len(item.title), len(item.summary), len(item.content), len(item.source_url), len(item.author), item.content_hash[:16],
        )
        try:
            info_item, created = info_repo.upsert_by_hash(
                title=item.title, summary=item.summary, content=item.content,
                source_url=item.source_url, source_type=item.source_type,
                author=item.author, published_at=item.published_at,
                language="zh", entities=[], topics=item.topics,
                raw_metadata=item.raw_metadata, content_hash=item.content_hash,
            )
            created_info += int(created)
            updated_info += int(not created)
        except Exception as exc:
            db.rollback()
            logger.warning("feed item upsert failed: title=%s err=%s", item.title[:80], str(exc)[:200])
            continue
        try:
            score = scorer.score(info_item, profile, feedback_stats, semantic_memories)
        except Exception:
            score = {"filtered": True}
        if score.get("filtered"):
            continue
        card = generate_feed_card(info_item, score, profile)
        card["info_item_id"] = info_item.id
        card["source_url"] = info_item.source_url or ""
        card["content_hash"] = info_item.content_hash or ""
        candidate_cards.append(card)

    return candidate_cards, created_info, updated_info


def refresh_feed(db: Session, user_id: int, limit: int | None = None, batch_id: str | None = None) -> dict:
    profile = _profile_with_defaults(ProfileRepository(db).get_or_create_default(user_id))
    total_limit = limit if limit else settings.feed_refresh_total_limit
    current_batch_id = batch_id or uuid.uuid4().hex[:12]
    now = datetime.now(UTC).replace(tzinfo=None)

    info_repo = InfoItemRepository(db)
    feed_repo = FeedRepository(db)
    feedback_stats = FeedFeedbackRepository(db).get_user_feedback_stats(user_id)
    semantic_memories = _safe_get_semantic_memories(user_id, db)
    scorer = FeedScorer()
    max_items = total_limit * 4

    # Daily dedup: don't create cards for info_items already used today
    existing_today = feed_repo.existing_info_item_ids_today(user_id)
    logger.info("feed daily dedup user=%d existing_today=%d", user_id, len(existing_today))

    all_source_stats: dict = {}
    total_created_info = total_updated_info = total_skipped = 0
    all_candidate_cards: list[dict] = []

    for retry_round in range(_MAX_RETRY_ROUNDS + 1):
        raw_items, source_stats = _run_async(SearchSourceManager().fetch_all())
        all_source_stats.update(source_stats)
        normalized = [item for item in (normalize_raw_item(raw) for raw in raw_items) if item]
        unique_items, skipped = deduplicate_items(normalized)
        total_skipped += skipped

        new_cards, ci, ui = _process_items_into_cards(
            unique_items, info_repo, profile, feedback_stats, semantic_memories, scorer, max_items, db,
        )
        total_created_info += ci
        total_updated_info += ui

        # Daily dedup: filter out items already used today
        before_filter = len(new_cards)
        new_cards = [c for c in new_cards if c.get("info_item_id") not in existing_today]
        if len(new_cards) < before_filter:
            logger.info("feed daily dedup filtered user=%d before=%d after=%d", user_id, before_filter, len(new_cards))

        all_candidate_cards.extend(new_cards)
        all_candidate_cards = _deduplicate_candidate_cards(all_candidate_cards)

        # Count by bucket
        bucket_counts: dict[str, int] = {}
        for card in all_candidate_cards:
            rt = card.get("relation_type", "far_domain")
            bucket_counts[rt] = bucket_counts.get(rt, 0) + 1

        missing = [b for b, t in _BATCH_TARGETS.items() if bucket_counts.get(b, 0) < t]
        if not missing or retry_round >= _MAX_RETRY_ROUNDS:
            break
        logger.info("feed retry round %d user=%d missing=%s", retry_round + 1, user_id, missing)

    # Mix with absolute targets
    mix_targets = {key: value for key, value in _BATCH_TARGETS.items()}
    mixed, bucket_info = mix_cards(all_candidate_cards, mix_targets, total_limit)

    is_complete = len(mixed) >= total_limit and not bool(bucket_info.get("missing", []))
    missing_buckets_list = bucket_info.get("missing", [])

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
            "is_complete_batch": is_complete,
            "batch_total": len(mixed),
            "missing_buckets": missing_buckets_list,
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
            "batch_id": current_batch_id,
            "generated_at": now,
        })
    created_cards: list = []
    if rows:
        try:
            created_cards = feed_repo.bulk_create(rows)
        except Exception as exc:
            db.rollback()
            logger.warning("feed card bulk_create failed: %s", str(exc)[:200])

    return {
        "created_info_items": total_created_info,
        "updated_info_items": total_updated_info,
        "created_feed_cards": len(created_cards),
        "skipped_duplicates": total_skipped,
        "source_stats": all_source_stats,
        "batch_id": current_batch_id,
        "bucket_counts": bucket_info.get("selected", {}),
        "missing_buckets": missing_buckets_list,
        "is_complete": is_complete,
        "failed_items": 0,
    }


def list_cards(db: Session, user_id: int, status: str | None = None, exposure_bucket: str | None = None, limit: int = 20, offset: int = 0, source_type: str | None = None, domain: str | None = None, all: bool = False) -> dict:
    profile = _profile_with_defaults(ProfileRepository(db).get_or_create_default(user_id))
    cards = FeedRepository(db).list_by_user(user_id, status, exposure_bucket, limit, offset, source_type, domain, latest_batch_only=not all)
    cards_list = [card_to_dict(card) for card in cards]
    batch_id = FeedRepository(db).latest_batch_id(user_id) if not all else None
    return {"cards": cards_list, "next_cursor": offset + len(cards) if len(cards) == limit else None, "mix": profile.feed_ratio_config, "batch_id": batch_id or ""}


def list_home_cards(db: Session, user_id: int) -> dict:
    count = settings.feed_home_card_count
    refresh_result = maybe_refresh_for_user(db, user_id, force=False)
    errors: list[str] = []

    # Get latest batch cards
    latest_batch = FeedRepository(db).latest_batch_id(user_id)
    if latest_batch:
        cards = FeedRepository(db).list_by_user(user_id, status=None, limit=count * 3, batch_id=latest_batch)
    else:
        cards = FeedRepository(db).list_by_user(user_id, status=None, limit=count)

    if not cards:
        try:
            force_result = refresh_feed(db, user_id, limit=settings.feed_refresh_total_limit, batch_id=uuid.uuid4().hex[:12])
            refresh_result = {"refreshed": True, "reason": "force_empty", "batch_id": force_result.get("batch_id"), "created_count": force_result.get("created_feed_cards", 0), "bucket_counts": force_result.get("bucket_counts", {}), "missing_buckets": force_result.get("missing_buckets", []), "is_complete": force_result.get("is_complete", False)}
            latest_batch = force_result.get("batch_id")
            if latest_batch:
                cards = FeedRepository(db).list_by_user(user_id, status=None, limit=count * 3, batch_id=latest_batch)
        except Exception as exc:
            errors.append(str(exc)[:300])

    return {
        "cards": [card_to_dict(card) for card in cards],
        "required_count": count,
        "is_complete": refresh_result.get("is_complete", len(cards) >= count),
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
    latest_batch = repo.latest_batch_id(user_id)
    all_cards = repo.list_by_user(user_id, limit=1000, include_hidden=True)
    saved = sum(1 for card in all_cards if card.status == "saved")
    hidden = sum(1 for card in all_cards if card.status == "ignored")
    avg_score = round(sum(card.final_score for card in all_cards) / len(all_cards), 4) if all_cards else 0

    current_batch_count = repo.count_by_batch(user_id, latest_batch) if latest_batch else 0
    bucket_dist = repo.stats_for_user(user_id, batch_id=latest_batch) if latest_batch else repo.stats_for_user(user_id)

    return {
        "cards_count": current_batch_count,
        "total_history_count": len(all_cards),
        "saved_count": saved,
        "hidden_count": hidden,
        "average_final_score": avg_score,
        "batch_id": latest_batch or "",
        "bucket_counts": bucket_dist.get("relation_type_distribution", {}),
    }


def card_to_dict(card) -> dict:
    detail = card.score_detail or {}
    score_fields = {"personal_relevance", "novelty", "cross_domain_distance", "opportunity_value", "source_credibility", "actionability", "final", "profile_match", "semantic_memory_match"}

    db_title = card.title or ""
    original_title = detail.get("original_title", "")
    source_type = detail.get("source_type", "")
    domain = detail.get("domain", "ai")

    if not original_title:
        original_title = db_title

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
        "batch_id": card.batch_id or "",
        "generated_at": card.generated_at.isoformat() if card.generated_at else None,
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
