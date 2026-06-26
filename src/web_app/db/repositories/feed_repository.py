from datetime import date, datetime

from sqlalchemy import func, select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import FeedCard, FeedFeedback


class FeedRepository(BaseRepository[FeedCard]):
    model = FeedCard

    def list_by_user(self, user_id: int, status: str | None = None, exposure_bucket: str | None = None, limit: int = 20, offset: int = 0, source_type: str | None = None, domain: str | None = None, include_hidden: bool = False, batch_id: str | None = None, today_only: bool = False, latest_batch_only: bool = False) -> list[FeedCard]:
        if latest_batch_only:
            batch_id = self.latest_batch_id(user_id)
        stmt = select(FeedCard).where(FeedCard.user_id == user_id).order_by(FeedCard.final_score.desc(), FeedCard.created_at.desc()).limit(limit).offset(offset)
        if not include_hidden:
            stmt = stmt.where(FeedCard.status != "ignored")
        if status:
            stmt = stmt.where(FeedCard.status == status)
        if exposure_bucket:
            stmt = stmt.where(FeedCard.exposure_bucket == exposure_bucket)
        if source_type:
            stmt = stmt.where(FeedCard.score_detail["source_type"].as_string() == source_type)
        if domain:
            stmt = stmt.where(FeedCard.score_detail["domain"].as_string() == domain)
        if batch_id:
            stmt = stmt.where(FeedCard.batch_id == batch_id)
        if today_only:
            today_start = datetime.combine(date.today(), datetime.min.time())
            stmt = stmt.where(FeedCard.generated_at >= today_start)
        return list(self.db.execute(stmt).scalars())

    def get_by_user(self, user_id: int, card_id: int) -> FeedCard | None:
        return self.db.execute(select(FeedCard).where(FeedCard.user_id == user_id, FeedCard.id == card_id)).scalar_one_or_none()

    def latest_batch_id(self, user_id: int) -> str | None:
        result = self.db.execute(
            select(FeedCard.batch_id)
            .where(FeedCard.user_id == user_id, FeedCard.batch_id.isnot(None))
            .group_by(FeedCard.batch_id)
            .order_by(func.max(FeedCard.created_at).desc())
            .limit(1)
        ).scalar_one_or_none()
        return result

    def existing_info_item_ids_today(self, user_id: int, only_complete_batches: bool = False) -> set[int]:
        """Return info_item_ids already used in today's feed cards.

        If only_complete_batches=True, only considers cards from complete batches
        (batches with >=5 cards across all 3 buckets). This prevents incomplete
        batches from poisoning the dedup pool for subsequent retry attempts.
        """
        today_start = datetime.combine(date.today(), datetime.min.time())
        if not only_complete_batches:
            rows = self.db.execute(
                select(FeedCard.info_item_id).where(
                    FeedCard.user_id == user_id,
                    FeedCard.created_at >= today_start,
                )
            ).scalars().all()
            return set(rows)

        # Find complete batch IDs for today
        # A complete batch has >=5 cards with explicit>=1, adjacent>=1, far>=1
        batch_rows = self.db.execute(
            select(FeedCard.batch_id, FeedCard.exposure_bucket, func.count(FeedCard.id)).where(
                FeedCard.user_id == user_id,
                FeedCard.created_at >= today_start,
                FeedCard.batch_id.isnot(None),
            ).group_by(FeedCard.batch_id, FeedCard.exposure_bucket)
        ).all()

        # Aggregate per-batch
        batch_buckets: dict[str, dict[str, int]] = {}
        for bid, bucket, cnt in batch_rows:
            if bid not in batch_buckets:
                batch_buckets[bid] = {}
            batch_buckets[bid][bucket] = cnt

        complete_batch_ids: set[str] = set()
        for bid, buckets in batch_buckets.items():
            total = sum(buckets.values())
            if (total >= 5 and buckets.get("explicit_related", 0) >= 1
                    and buckets.get("adjacent_domain", 0) >= 1
                    and buckets.get("far_domain", 0) >= 1):
                complete_batch_ids.add(bid)

        if not complete_batch_ids:
            return set()

        rows = self.db.execute(
            select(FeedCard.info_item_id).where(
                FeedCard.user_id == user_id,
                FeedCard.created_at >= today_start,
                FeedCard.batch_id.in_(complete_batch_ids),
            )
        ).scalars().all()
        return set(rows)

    def bulk_create(self, rows: list[dict]) -> list[FeedCard]:
        import logging
        _logger = logging.getLogger(__name__)
        cards = []
        failed = 0
        for i, row in enumerate(rows):
            try:
                cards.append(FeedCard(**row))
            except Exception:
                failed += 1
                _logger.exception("feed bulk_create card construction failed row=%s keys=%s", i, list(row.keys())[:10])
        if failed:
            _logger.error("feed bulk_create construction failures: %s/%s cards could not be built", failed, len(rows))
        self.db.add_all(cards)
        self._commit()
        for card in cards:
            self.db.refresh(card)
        if len(cards) != len(rows):
            _logger.error("feed bulk_create mismatch input=%s constructed=%s", len(rows), len(cards))
        return cards

    def bucket_counts_for_batch(self, user_id: int, batch_id: str) -> dict:
        """Return {explicit_related: N, adjacent_domain: N, far_domain: N, total: N} for a batch."""
        rows = self.db.execute(
            select(FeedCard.exposure_bucket, func.count(FeedCard.id)).where(
                FeedCard.user_id == user_id,
                FeedCard.batch_id == batch_id,
            ).group_by(FeedCard.exposure_bucket)
        ).all()
        counts = dict(rows)
        total = sum(counts.values())
        return {
            "explicit_related": counts.get("explicit_related", 0),
            "adjacent_domain": counts.get("adjacent_domain", 0),
            "far_domain": counts.get("far_domain", 0),
            "total": total,
        }

    def count_by_batch(self, user_id: int, batch_id: str) -> int:
        return self.db.execute(
            select(func.count(FeedCard.id)).where(FeedCard.user_id == user_id, FeedCard.batch_id == batch_id)
        ).scalar_one()

    def stats_for_user(self, user_id: int, batch_id: str | None = None) -> dict:
        stmt = select(FeedCard.exposure_bucket, func.count(FeedCard.id)).where(FeedCard.user_id == user_id)
        if batch_id:
            stmt = stmt.where(FeedCard.batch_id == batch_id)
        rows = self.db.execute(stmt.group_by(FeedCard.exposure_bucket)).all()
        source_stmt = select(FeedCard.score_detail["source_type"].as_string(), func.count(FeedCard.id)).where(FeedCard.user_id == user_id)
        if batch_id:
            source_stmt = source_stmt.where(FeedCard.batch_id == batch_id)
        source_rows = self.db.execute(source_stmt.group_by(FeedCard.score_detail["source_type"].as_string())).all()
        return {"relation_type_distribution": dict(rows), "source_type_distribution": dict(source_rows)}


class FeedFeedbackRepository(BaseRepository[FeedFeedback]):
    model = FeedFeedback

    def create_feedback(self, user_id: int, card_id: int, action: str, metadata: dict | None = None) -> FeedFeedback:
        return self.create(user_id=user_id, card_id=card_id, action=action, metadata_json=metadata or {})

    def get_user_feedback_stats(self, user_id: int) -> dict:
        rows = self.db.execute(select(FeedFeedback.action, func.count(FeedFeedback.id)).where(FeedFeedback.user_id == user_id).group_by(FeedFeedback.action)).all()
        return {"actions": dict(rows), "positive_topics": {}, "negative_topics": {}}
