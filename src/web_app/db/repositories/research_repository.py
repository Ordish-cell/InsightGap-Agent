from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import ResearchRun


class ResearchRunRepository(BaseRepository[ResearchRun]):
    model = ResearchRun

    def get_by_user(self, user_id: int, run_id: str) -> ResearchRun | None:
        return self.db.execute(select(ResearchRun).where(ResearchRun.user_id == user_id, ResearchRun.id == run_id)).scalar_one_or_none()

    def list_by_user(self, user_id: int, limit: int = 20, offset: int = 0) -> list[ResearchRun]:
        stmt = select(ResearchRun).where(ResearchRun.user_id == user_id).order_by(ResearchRun.created_at.desc()).limit(limit).offset(offset)
        return list(self.db.execute(stmt).scalars())
