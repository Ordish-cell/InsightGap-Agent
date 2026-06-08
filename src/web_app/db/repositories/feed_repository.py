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

    def existing_info_item_ids_today(self, user_id: int) -> set[int]:
        today_start = datetime.combine(date.today(), datetime.min.time())
        rows = self.db.execute(
            select(FeedCard.info_item_id).where(
                FeedCard.user_id == user_id,
                FeedCard.created_at >= today_start,
            )
        ).scalars().all()
        return set(rows)

    def bulk_create(self, rows: list[dict]) -> list[FeedCard]:
        cards = [FeedCard(**row) for row in rows]
        self.db.add_all(cards)
        self.db.commit()
        for card in cards:
            self.db.refresh(card)
        return cards

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
