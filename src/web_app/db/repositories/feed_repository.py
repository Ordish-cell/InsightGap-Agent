from sqlalchemy import func, select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import FeedCard, FeedFeedback


class FeedRepository(BaseRepository[FeedCard]):
    model = FeedCard

    def list_by_user(self, user_id: int, status: str | None = None, exposure_bucket: str | None = None, limit: int = 20, offset: int = 0, source_type: str | None = None, domain: str | None = None, include_hidden: bool = False) -> list[FeedCard]:
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
        return list(self.db.execute(stmt).scalars())

    def get_by_user(self, user_id: int, card_id: int) -> FeedCard | None:
        return self.db.execute(select(FeedCard).where(FeedCard.user_id == user_id, FeedCard.id == card_id)).scalar_one_or_none()

    def bulk_create(self, rows: list[dict]) -> list[FeedCard]:
        cards = [FeedCard(**row) for row in rows]
        self.db.add_all(cards)
        self.db.commit()
        for card in cards:
            self.db.refresh(card)
        return cards

    def stats_for_user(self, user_id: int) -> dict:
        rows = self.db.execute(select(FeedCard.exposure_bucket, func.count(FeedCard.id)).where(FeedCard.user_id == user_id).group_by(FeedCard.exposure_bucket)).all()
        source_rows = self.db.execute(select(FeedCard.score_detail["source_type"].as_string(), func.count(FeedCard.id)).where(FeedCard.user_id == user_id).group_by(FeedCard.score_detail["source_type"].as_string())).all()
        return {"relation_type_distribution": dict(rows), "source_type_distribution": dict(source_rows)}


class FeedFeedbackRepository(BaseRepository[FeedFeedback]):
    model = FeedFeedback

    def create_feedback(self, user_id: int, card_id: int, action: str, metadata: dict | None = None) -> FeedFeedback:
        return self.create(user_id=user_id, card_id=card_id, action=action, metadata_json=metadata or {})

    def get_user_feedback_stats(self, user_id: int) -> dict:
        rows = self.db.execute(select(FeedFeedback.action, func.count(FeedFeedback.id)).where(FeedFeedback.user_id == user_id).group_by(FeedFeedback.action)).all()
        return {"actions": dict(rows), "positive_topics": {}, "negative_topics": {}}
