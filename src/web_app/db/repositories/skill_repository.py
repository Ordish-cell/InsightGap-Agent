from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Skill


class SkillRepository(BaseRepository[Skill]):
    model = Skill

    def list_by_user(self, user_id: int) -> list[Skill]:
        return list(self.db.execute(select(Skill).where(Skill.user_id == user_id).order_by(Skill.created_at.desc())).scalars())

    def get_by_user(self, user_id: int, skill_id: int) -> Skill | None:
        return self.db.execute(select(Skill).where(Skill.user_id == user_id, Skill.id == skill_id)).scalar_one_or_none()
