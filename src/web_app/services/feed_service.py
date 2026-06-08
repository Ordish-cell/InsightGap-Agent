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
from src.web_app.feed.sources.bucket_seed import BucketSeedSource
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


def get_latest_batch_bucket_counts(db: Session, user_id: int) -> dict:
    repo = FeedRepository(db)
    latest = repo.latest_batch_id(user_id)
    if not latest:
        return {"explicit_related": 0, "adjacent_domain": 0, "far_domain": 0, "total": 0, "batch_id": None}
    counts = repo.bucket_counts_for_batch(user_id, latest)
    counts["batch_id"] = latest
    return counts


def is_complete_feed_batch(counts: dict) -> bool:
    return (
        counts.get("total", 0) >= 5
        and counts.get("explicit_related", 0) >= 1
        and counts.get("adjacent_domain", 0) >= 1
        and counts.get("far_domain", 0) >= 1
    )


def _canonicalize_url(url: str | None) -> str:
    if not url:
        return ""
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return ""
    query_parts = parts.query.split("&")
    clean_query = "&".join(p for p in query_parts if p and not p.startswith("utm_"))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), clean_query, ""))


def _normalize_title_key(title: str) -> str:
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:120]


def _is_today(dt: datetime | None) -> bool:
    if not dt:
        return False
    return dt.date() == date.today()


def _deduplicate_candidate_cards(cards: list[dict]) -> list[dict]:
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
                continue
            seen_item_ids.add(item_id)

        source_url = _canonicalize_url(str(card.get("source_url", "")) or str((card.get("evidence") or [{}])[0].get("url", "")))
        if source_url:
            if source_url in seen_urls:
                continue
            seen_urls.add(source_url)

        title_key = _normalize_title_key(str(card.get("original_title", card.get("title", ""))))
        if title_key:
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

        result.append(card)

    removed = before - len(result)
    if removed:
        logger.info("feed dedup batch: before=%d after=%d removed=%d", before, len(result), removed)
    return result


def _acquire_refresh_lock(db: Session, user_id: int) -> bool:
    lock_id = hash(f"feed_refresh:{user_id}") % (2 ** 31)
    result = db.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": lock_id}).scalar()
    return bool(result)


def _release_refresh_lock(db: Session, user_id: int) -> None:
    lock_id = hash(f"feed_refresh:{user_id}") % (2 ** 31)
    db.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": lock_id})


def _count_cards_by_bucket(cards: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"explicit_related": 0, "adjacent_domain": 0, "far_domain": 0}
    for card in cards:
        rt = card.get("relation_type", "far_domain")
        if rt in counts:
            counts[rt] += 1
        else:
            counts[rt] = 1
    return counts


# ── full seed fallback: generates cards from seeds when real search fails ──

def _seed_cards_for_bucket(
    db: Session,
    bucket: str,
    need: int,
    info_repo: InfoItemRepository,
    profile: Any,
    feedback_stats: dict,
    semantic_memories: list,
    scorer: FeedScorer,
    existing_today: set[int],
) -> list[dict]:
    """Fetch BucketSeedSource items for a bucket and convert to candidate cards."""
    seed_source = BucketSeedSource(buckets=[bucket])
    try:
        seed_raw = _run_async(seed_source.fetch())
    except Exception as exc:
        logger.error("feed seed fetch failed bucket=%s: %s", bucket, str(exc)[:200])
        return []

    cards: list[dict] = []
    for raw in seed_raw:
        if len(cards) >= need:
            break
        normalized = normalize_raw_item(raw)
        if not normalized:
            continue
        normalized.raw_metadata["source_kind"] = "bucket_seed"
        normalized.raw_metadata["search_bucket"] = bucket
        try:
            info_item, _ = info_repo.upsert_by_hash(
                title=normalized.title, summary=normalized.summary, content=normalized.content,
                source_url=normalized.source_url, source_type=normalized.source_type,
                author=normalized.author, published_at=normalized.published_at,
                language="zh", entities=[], topics=normalized.topics,
                raw_metadata=normalized.raw_metadata, content_hash=normalized.content_hash,
            )
        except Exception as exc:
            db.rollback()
            logger.warning("feed seed upsert failed: title=%s err=%s", raw.title[:80], str(exc)[:200])
            continue

        if info_item.id in existing_today:
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
        card["relation_type"] = bucket
        card["exposure_bucket"] = bucket
        card["source_kind"] = "bucket_seed"
        card["provider"] = "bucket_seed"
        cards.append(card)

    return cards


def full_seed_fallback(
    db: Session,
    info_repo: InfoItemRepository,
    profile: Any,
    feedback_stats: dict,
    semantic_memories: list,
    scorer: FeedScorer,
    existing_today: set[int],
    total_limit: int,
) -> list[dict]:
    """Generate cards entirely from seeds: explicit=2, adjacent=2, far=1."""
    logger.warning("feed full_seed_fallback START: real search produced 0 cards — using seeds")
    all_seed_cards: list[dict] = []

    for bucket, need in _BATCH_TARGETS.items():
        cards = _seed_cards_for_bucket(
            db, bucket, need, info_repo, profile, feedback_stats,
            semantic_memories, scorer, existing_today,
        )
        if len(cards) < need:
            logger.warning("feed full_seed_fallback bucket=%s needed=%d got=%d", bucket, need, len(cards))
        else:
            logger.warning("feed full_seed_fallback bucket=%s got=%d", bucket, len(cards))
        all_seed_cards.extend(cards)

    all_seed_cards = _deduplicate_candidate_cards(all_seed_cards)
    counts = _count_cards_by_bucket(all_seed_cards)
    logger.warning("feed full_seed_fallback DONE: total=%d counts=%s", len(all_seed_cards), counts)
    return all_seed_cards


# ── ensure bucket minimums (called during refresh_feed) ──

def ensure_bucket_minimums(
    db: Session,
    cards: list[dict],
    info_repo: InfoItemRepository,
    profile: Any,
    feedback_stats: dict,
    semantic_memories: list,
    scorer: FeedScorer,
    existing_today: set[int],
) -> list[dict]:
    """Inject BucketSeedSource cards for buckets below minimum targets."""
    bucket_counts = _count_cards_by_bucket(cards)
    logger.warning("feed ensure bucket minimums before=%s", bucket_counts)

    for bucket, target in _BATCH_TARGETS.items():
        current = bucket_counts.get(bucket, 0)
        needed = max(0, target - current)
        if needed <= 0:
            continue

        logger.warning("feed injecting bucket seeds bucket=%s needed=%d", bucket, needed)
        if bucket == "far_domain":
            logger.warning("far slot filled by bucket_seed because no valid real far result")
        seed_cards = _seed_cards_for_bucket(
            db, bucket, needed, info_repo, profile, feedback_stats,
            semantic_memories, scorer, existing_today,
        )
        logger.warning("feed seed cards generated bucket=%s count=%d", bucket, len(seed_cards))
        cards = cards + seed_cards

    cards = _deduplicate_candidate_cards(cards)
    bucket_counts_after = _count_cards_by_bucket(cards)
    logger.warning("feed ensure bucket minimums after=%s", bucket_counts_after)
    return cards


# ── main refresh_feed (with full try/except, always logs completed/failed) ──

def refresh_feed(db: Session, user_id: int, limit: int | None = None, batch_id: str | None = None) -> dict:
    profile = _profile_with_defaults(ProfileRepository(db).get_or_create_default(user_id))
    total_limit = limit if limit else settings.feed_refresh_total_limit
    current_batch_id = batch_id or uuid.uuid4().hex[:12]
    now = datetime.now(UTC).replace(tzinfo=None)

    logger.warning("refresh_feed started user=%s limit=%s batch_id=%s", user_id, total_limit, current_batch_id)

    info_repo = InfoItemRepository(db)
    feed_repo = FeedRepository(db)
    feedback_stats = FeedFeedbackRepository(db).get_user_feedback_stats(user_id)
    semantic_memories = _safe_get_semantic_memories(user_id, db)
    scorer = FeedScorer()
    max_items = total_limit * 4

    existing_today = feed_repo.existing_info_item_ids_today(user_id, only_complete_batches=True)

    all_candidate_cards: list[dict] = []
    created_count = 0
    source_summary: dict = {}
    failed_items = 0

    try:
        from src.web_app.feed.sources.manager import SearchSourceManager

        all_source_stats: dict = {}
        total_created_info = total_updated_info = total_skipped = 0

        for retry_round in range(_MAX_RETRY_ROUNDS + 1):
            raw_items, source_stats, source_summary = _run_async(SearchSourceManager().fetch_all())
            all_source_stats.update(source_stats)
            normalized = [item for item in (normalize_raw_item(raw) for raw in raw_items) if item]
            unique_items, skipped = deduplicate_items(normalized)
            total_skipped += skipped

            new_cards, ci, ui = _process_items_into_cards(
                unique_items, info_repo, profile, feedback_stats, semantic_memories, scorer, max_items, db,
            )
            total_created_info += ci
            total_updated_info += ui

            new_cards = [c for c in new_cards if c.get("info_item_id") not in existing_today]
            all_candidate_cards.extend(new_cards)
            all_candidate_cards = _deduplicate_candidate_cards(all_candidate_cards)

            bucket_counts = _count_cards_by_bucket(all_candidate_cards)
            missing = [b for b, t in _BATCH_TARGETS.items() if bucket_counts.get(b, 0) < t]

            if missing:
                all_candidate_cards = ensure_bucket_minimums(
                    db, all_candidate_cards, info_repo, profile, feedback_stats,
                    semantic_memories, scorer, existing_today,
                )
                bucket_counts = _count_cards_by_bucket(all_candidate_cards)
                missing = [b for b, t in _BATCH_TARGETS.items() if bucket_counts.get(b, 0) < t]

            if not missing or retry_round >= _MAX_RETRY_ROUNDS:
                break
            logger.info("feed retry round %d user=%d missing=%s", retry_round + 1, user_id, missing)

        # ── Step 1: log candidate cards ──
        pre_patch_counts = _count_cards_by_bucket(all_candidate_cards)
        logger.warning("feed candidate_cards count=%s buckets=%s", len(all_candidate_cards), pre_patch_counts)

        # ── MANDATORY: always patch missing buckets with seeds before final mix ──
        missing_after_loop = [b for b, t in _BATCH_TARGETS.items() if pre_patch_counts.get(b, 0) < t]
        if missing_after_loop:
            logger.warning("feed seed patch required user=%s missing=%s before=%s", user_id, missing_after_loop, pre_patch_counts)
            all_candidate_cards = ensure_bucket_minimums(
                db, all_candidate_cards, info_repo, profile, feedback_stats,
                semantic_memories, scorer, existing_today,
            )

        # ── Step 2: log after ensure_minimums ──
        post_patch_counts = _count_cards_by_bucket(all_candidate_cards)
        logger.warning("feed after ensure_minimums count=%s buckets=%s", len(all_candidate_cards), post_patch_counts)

        # If still missing buckets after seed patch, use full_seed_fallback as last resort
        missing_after_patch = [b for b, t in _BATCH_TARGETS.items() if post_patch_counts.get(b, 0) < t]
        if missing_after_patch:
            logger.warning("feed seed patch incomplete user=%s still_missing=%s — triggering full_seed_fallback", user_id, missing_after_patch)
            fallback_cards = full_seed_fallback(
                db, info_repo, profile, feedback_stats, semantic_memories,
                scorer, existing_today, total_limit,
            )
            all_candidate_cards = all_candidate_cards + fallback_cards
            all_candidate_cards = _deduplicate_candidate_cards(all_candidate_cards)
            post_fallback_counts = _count_cards_by_bucket(all_candidate_cards)
            still_short = [b for b, t in _BATCH_TARGETS.items() if post_fallback_counts.get(b, 0) < t]
            if still_short:
                logger.error("feed CRITICAL: buckets still missing after full_seed_fallback user=%s missing=%s", user_id, still_short)
                failed_items = 1

        # Mix with hard targets
        mix_targets = {key: value for key, value in _BATCH_TARGETS.items()}
        mixed, bucket_info = mix_cards(all_candidate_cards, mix_targets, total_limit)

        # ── Step 3: log mixed result ──
        mixed_counts = _count_cards_by_bucket(mixed)
        logger.warning("feed mixed_cards count=%s buckets=%s bucket_info=%s", len(mixed), mixed_counts, bucket_info)

        # ── Section 6: final hard guarantee — never write only explicit if adjacent/far missing ──
        actual_mixed_counts = _count_cards_by_bucket(mixed)
        missing_from_mix = [b for b, t in _BATCH_TARGETS.items() if actual_mixed_counts.get(b, 0) < t]
        if missing_from_mix:
            logger.error(
                "feed mixed_cards missing buckets before create user=%s missing=%s counts=%s — applying final seed patch",
                user_id, missing_from_mix, actual_mixed_counts,
            )
            # Build seed cards directly for missing buckets, bypassing scorer filtering
            for missing_bucket in missing_from_mix:
                need = _BATCH_TARGETS[missing_bucket] - actual_mixed_counts.get(missing_bucket, 0)
                if need <= 0:
                    continue
                seed_cards = _seed_cards_for_bucket(
                    db, missing_bucket, need, info_repo, profile, feedback_stats,
                    semantic_memories, scorer, existing_today,
                )
                logger.warning("feed final patch bucket=%s need=%d got=%d", missing_bucket, need, len(seed_cards))
                mixed.extend(seed_cards)

            # Re-dedup and re-count
            mixed = _deduplicate_candidate_cards(mixed)
            actual_mixed_counts = _count_cards_by_bucket(mixed)
            still_missing = [b for b, t in _BATCH_TARGETS.items() if actual_mixed_counts.get(b, 0) < t]
            if still_missing:
                logger.critical(
                    "feed CANNOT CREATE VIABLE BATCH user=%s still_missing=%s counts=%s — refusing to write only explicit",
                    user_id, still_missing, actual_mixed_counts,
                )
            logger.warning("feed after final patch count=%s buckets=%s", len(mixed), actual_mixed_counts)

        is_complete = bucket_info.get("is_complete", False) and len(mixed) >= total_limit
        # Also update is_complete based on actual mixed counts
        still_short_any = [b for b, t in _BATCH_TARGETS.items() if actual_mixed_counts.get(b, 0) < t]
        if still_short_any:
            is_complete = False
        missing_buckets_list = sorted(set(bucket_info.get("missing", []) + still_short_any))

        # ── Section 4: unify bucket field before ORM mapping ──
        rows = []
        for card in mixed:
            # Unify bucket from multiple possible field names
            bucket = (
                card.get("exposure_bucket")
                or card.get("relation_type")
                or card.get("search_bucket")
                or card.get("score", {}).get("relation_type")
            )
            if bucket not in ("explicit_related", "adjacent_domain", "far_domain"):
                logger.error("feed invalid bucket for card title=%s bucket=%s — skipping", card.get("title", "")[:80], bucket)
                continue

            card["exposure_bucket"] = bucket
            card["relation_type"] = bucket

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
                "source_kind": card.get("source_kind", ""),
                "provider": card.get("provider", ""),
                "search_bucket": card.get("search_bucket", bucket),
            })

            # ── Step 4: log each card before create ──
            logger.warning(
                "feed pre-create card bucket=%s title=%s info_item_id=%s source_url=%s source_kind=%s provider=%s final_score=%s",
                bucket,
                (card.get("title") or "")[:80],
                card.get("info_item_id"),
                (card.get("source_url") or "")[:100],
                card.get("source_kind", ""),
                card.get("provider", ""),
                card.get("final_score", 0),
            )

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
                "exposure_bucket": bucket,
                "status": "active",
                "batch_id": current_batch_id,
                "generated_at": now,
            })

        # ── Step 5: bulk_create with mismatch detection ──
        created_count = 0
        if rows:
            try:
                created_cards = feed_repo.bulk_create(rows)
                created_count = len(created_cards)
                logger.warning("feed created_cards count=%s ids=%s", created_count, [c.id for c in created_cards])
                if created_count != len(rows):
                    logger.error(
                        "feed bulk_create mismatch input=%s created=%s missing=%s",
                        len(rows), created_count, len(rows) - created_count,
                    )
            except Exception as exc:
                db.rollback()
                logger.exception("feed card bulk_create failed: %s", str(exc)[:200])

        # ── Step 6: verify DB after create ──
        try:
            db_counts = feed_repo.bucket_counts_for_batch(user_id, current_batch_id)
            logger.warning("feed db_counts after create batch_id=%s counts=%s", current_batch_id, db_counts)
        except Exception:
            logger.warning("feed db_counts verification failed", exc_info=True)

        logger.warning(
            "refresh_feed completed user=%s batch_id=%s created_count=%s bucket_counts=%s is_complete=%s failed_items=%s",
            user_id, current_batch_id, created_count, actual_mixed_counts, is_complete, failed_items,
        )

        return {
            "batch_id": current_batch_id,
            "created_count": created_count,
            "bucket_counts": actual_mixed_counts,
            "missing_buckets": missing_buckets_list,
            "is_complete": is_complete,
            "failed_items": failed_items,
            "source_summary": source_summary if source_summary else {},
            "created_info_items": total_created_info,
            "updated_info_items": total_updated_info,
            "created_feed_cards": created_count,
            "skipped_duplicates": total_skipped,
            "source_stats": all_source_stats,
        }
    except Exception as exc:
        db.rollback()
        logger.exception("refresh_feed failed user=%s batch_id=%s error=%s", user_id, current_batch_id, str(exc)[:300])
        return {
            "batch_id": current_batch_id,
            "created_count": 0,
            "bucket_counts": {},
            "missing_buckets": ["explicit_related", "adjacent_domain", "far_domain"],
            "is_complete": False,
            "failed": True,
            "error": str(exc)[:500],
            "source_summary": {},
            "created_feed_cards": 0,
            "source_stats": {},
            "failed_items": 1,
        }


# ── maybe_refresh_for_user (P1: deferred attempt_at, retry on 0 cards) ──

def maybe_refresh_for_user(db: Session, user_id: int, force: bool = False) -> dict[str, Any]:
    """Trigger feed refresh — at most once per day unless force=True.

    P1 fix: attempt_at is only set AFTER refresh_feed produces cards > 0.
    If attempted_today but no batch exists, allow retry.
    """
    profile_repo = ProfileRepository(db)
    profile = profile_repo.get_by_user(user_id)
    if not profile:
        logger.warning("maybe_refresh_for_user user=%s no_profile", user_id)
        return {"refreshed": False, "skipped": True, "reason": "no_profile", "batch_id": None, "created_count": 0, "bucket_counts": {}, "missing_buckets": []}

    attempted_today = _is_today(profile.last_feed_refresh_attempt_at)
    refreshed_today = _is_today(profile.last_feed_refreshed_at)

    # Check if any batch actually exists
    existing_batch_id = FeedRepository(db).latest_batch_id(user_id)
    has_any_batch = bool(existing_batch_id)

    # Allow retry if attempted today but no batch exists (refresh crashed / produced 0)
    if not force and attempted_today and has_any_batch:
        action = "skip_already_attempted_today"
        logger.warning(
            "feed refresh decision user=%s force=%s attempted_today=%s refreshed_today=%s has_batch=%s action=%s",
            user_id, force, attempted_today, refreshed_today, has_any_batch, action,
        )
        counts = FeedRepository(db).bucket_counts_for_batch(user_id, existing_batch_id) if existing_batch_id else {}
        return {
            "refreshed": False, "skipped": True, "reason": "already_attempted_today",
            "batch_id": existing_batch_id, "created_count": 0,
            "bucket_counts": {k: v for k, v in counts.items() if k != "batch_id"},
            "missing_buckets": [],
        }

    if not force and attempted_today and not has_any_batch:
        logger.warning(
            "feed refresh decision user=%s force=%s attempted_today=True but no batch exists — retrying",
            user_id, force,
        )

    # Acquire lock first, don't commit attempt_at yet
    locked = _acquire_refresh_lock(db, user_id)
    if not locked:
        action = "skip_locked"
        logger.warning(
            "feed refresh decision user=%s force=%s attempted_today=%s refreshed_today=%s action=%s",
            user_id, force, attempted_today, refreshed_today, action,
        )
        return {"refreshed": False, "skipped": True, "reason": "refresh_in_progress", "batch_id": None, "created_count": 0, "bucket_counts": {}, "missing_buckets": []}

    action = "execute_refresh"
    logger.warning(
        "feed refresh decision user=%s force=%s attempted_today=%s refreshed_today=%s has_batch=%s action=%s",
        user_id, force, attempted_today, refreshed_today, has_any_batch, action,
    )

    batch_id = uuid.uuid4().hex[:12]
    try:
        result = refresh_feed(db, user_id, limit=settings.feed_refresh_total_limit, batch_id=batch_id)

        created_count = result.get("created_count", 0)
        is_c = result.get("is_complete", False)
        batch_count = FeedRepository(db).count_by_batch(user_id, batch_id) if batch_id else 0
        bucket_counts_result = result.get("bucket_counts", {})

        # Only mark attempt done if we have a minimally viable batch:
        # ≥3 cards AND at least explicit/adjacent/far each present
        has_min_buckets = (
            bucket_counts_result.get("explicit_related", 0) >= 1
            and bucket_counts_result.get("adjacent_domain", 0) >= 1
            and bucket_counts_result.get("far_domain", 0) >= 1
        )
        viable = (created_count >= 3 and has_min_buckets) or is_c
        if viable:
            profile.last_feed_refresh_attempt_at = datetime.now(UTC).replace(tzinfo=None)
            logger.warning("feed refresh attempt marked done user=%s created_count=%s buckets=%s", user_id, created_count, bucket_counts_result)
        else:
            logger.warning("feed refresh batch not viable (created=%s, buckets=%s); NOT marking attempt done so next request can retry", created_count, bucket_counts_result)

        if is_c:
            profile.last_feed_refreshed_at = datetime.now(UTC).replace(tzinfo=None)

        db.commit()

        return {
            "refreshed": True,
            "reason": "force" if force else "daily",
            "batch_id": batch_id,
            "created_count": created_count,
            "bucket_counts": result.get("bucket_counts", {}),
            "missing_buckets": result.get("missing_buckets", []),
            "is_complete": is_c,
            "source_stats": result.get("source_stats", {}),
            "source_summary": result.get("source_summary", {}),
        }
    except Exception as exc:
        db.rollback()
        logger.exception("feed refresh crashed user=%s batch_id=%s — NOT marking attempt done", user_id, batch_id)
        return {
            "refreshed": False, "skipped": False, "failed": True,
            "reason": "refresh_crashed", "error": str(exc)[:300],
            "batch_id": None, "created_count": 0,
            "bucket_counts": {}, "missing_buckets": [],
            "is_complete": False,
        }
    finally:
        if locked:
            try:
                _release_refresh_lock(db, user_id)
            except Exception:
                logger.warning("release feed refresh lock failed", exc_info=True)


# ── list_home_cards (P1: zero-cards retry) ──

def list_home_cards(db: Session, user_id: int) -> dict:
    """Read-mostly: auto-refresh once per day at most, then read latest batch.
    P1: if batch is empty (0 cards), trigger one force retry.
    """
    count = settings.feed_home_card_count
    feed_repo = FeedRepository(db)

    # Attempt daily first refresh (no-op if already attempted today with existing batch)
    refresh_result = maybe_refresh_for_user(db, user_id, force=False)

    # Read latest batch
    latest = feed_repo.latest_batch_id(user_id)
    if latest:
        counts = feed_repo.bucket_counts_for_batch(user_id, latest)
        cards = feed_repo.list_by_user(user_id, status=None, limit=count * 3, batch_id=latest)
    else:
        counts = {"explicit_related": 0, "adjacent_domain": 0, "far_domain": 0, "total": 0}
        cards = []

    # P1: if 0 cards and no refresh was attempted (or refresh was skipped), force retry once
    if counts.get("total", 0) == 0 and not refresh_result.get("refreshed"):
        logger.warning("feed home found zero cards, forcing one retry user=%s", user_id)
        # Only retry if we haven't already attempted today WITH a successful batch
        profile = ProfileRepository(db).get_by_user(user_id)
        attempted_today = _is_today(profile.last_feed_refresh_attempt_at) if profile else False
        has_any_batch = bool(latest)
        if not (attempted_today and has_any_batch):
            refresh_result = maybe_refresh_for_user(db, user_id, force=True)
            latest = feed_repo.latest_batch_id(user_id)
            if latest:
                counts = feed_repo.bucket_counts_for_batch(user_id, latest)
                cards = feed_repo.list_by_user(user_id, status=None, limit=count * 3, batch_id=latest)

    card_dicts = [card_to_dict(card) for card in cards]
    is_c = is_complete_feed_batch(counts)
    missing_buckets = [] if is_c else [b for b, t in _BATCH_TARGETS.items() if counts.get(b, 0) < t]

    logger.warning(
        "feed home read user=%s batch_id=%s counts=%s is_complete=%s card_count=%s",
        user_id, latest, {k: v for k, v in counts.items() if k != "batch_id"}, is_c, len(card_dicts),
    )

    return {
        "cards": card_dicts,
        "required_count": count,
        "batch_id": latest or "",
        "is_complete": is_c,
        "bucket_counts": counts,
        "missing_buckets": missing_buckets,
        "refresh_result": refresh_result,
        "errors": [],
    }


def list_cards(db: Session, user_id: int, status: str | None = None, exposure_bucket: str | None = None, limit: int = 20, offset: int = 0, source_type: str | None = None, domain: str | None = None, all: bool = False) -> dict:
    """Read-only: list feed cards for a user. Never triggers refresh."""
    profile = _profile_with_defaults(ProfileRepository(db).get_or_create_default(user_id))
    feed_repo = FeedRepository(db)

    latest_batch = feed_repo.latest_batch_id(user_id)
    batch_id = None
    is_c = None
    bucket_counts = None
    missing = None

    if not all and latest_batch:
        batch_id = latest_batch
        counts = feed_repo.bucket_counts_for_batch(user_id, latest_batch)
        is_c = is_complete_feed_batch(counts)
        bucket_counts = counts
        if not is_c:
            missing = [b for b, t in _BATCH_TARGETS.items() if counts.get(b, 0) < t]

    cards = feed_repo.list_by_user(user_id, status, exposure_bucket, limit, offset, source_type, domain, batch_id=batch_id)
    cards_list = [card_to_dict(card) for card in cards]

    logger.warning(
        "feed cards read user=%s batch_id=%s all=%s count=%s is_complete=%s",
        user_id, batch_id, all, len(cards_list), is_c,
    )

    return {
        "cards": cards_list,
        "next_cursor": offset + len(cards) if len(cards) == limit else None,
        "mix": profile.feed_ratio_config,
        "batch_id": batch_id or "",
        "is_complete": is_c,
        "bucket_counts": bucket_counts,
        "missing_buckets": missing,
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
        "source_kind": detail.get("source_kind", ""),
        "provider": detail.get("provider", ""),
        "search_query": detail.get("search_query", ""),
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


def _process_items_into_cards(
    unique_items: list, info_repo: InfoItemRepository, profile: Any,
    feedback_stats: dict, semantic_memories: list, scorer: FeedScorer, max_items: int,
    db: Session,
) -> tuple[list[dict], int, int]:
    candidate_cards: list[dict[str, Any]] = []
    created_info = updated_info = 0
    for item in unique_items[:max_items * 3]:
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
        except Exception:
            db.rollback()
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
        meta = info_item.raw_metadata or {}
        card["source_kind"] = meta.get("source_kind", "")
        card["provider"] = meta.get("provider", "")
        card["search_query"] = meta.get("search_query", "")
        card["search_bucket"] = meta.get("search_bucket", "")
        candidate_cards.append(card)
    return candidate_cards, created_info, updated_info


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
