from sqlalchemy import func, select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Memory


class MemoryRepository(BaseRepository[Memory]):
    model = Memory

    def list_by_user(self, user_id: int) -> list[Memory]:
        return list(self.db.execute(select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())).scalars())

    def search(self, user_id: int, query: str = "", memory_type: str | None = None, min_importance: float = 0.0) -> list[Memory]:
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.importance >= min_importance).order_by(Memory.importance.desc(), Memory.created_at.desc())
        if query:
            stmt = stmt.where(Memory.content.like(f"%{query}%"))
        if memory_type:
            stmt = stmt.where(Memory.memory_type == memory_type)
        return list(self.db.execute(stmt).scalars())

    def search_by_type(self, user_id: int, memory_type: str = "semantic", min_importance: float = 0.0) -> list[Memory]:
        stmt = select(Memory).where(
            Memory.user_id == user_id,
            Memory.memory_type == memory_type,
            Memory.importance >= min_importance,
        ).order_by(Memory.importance.desc())
        return list(self.db.execute(stmt).scalars())

    def counts_by_type(self, user_id: int) -> list[tuple[str, int, float]]:
        stmt = select(Memory.memory_type, func.count(Memory.id), func.avg(Memory.importance)).where(Memory.user_id == user_id).group_by(Memory.memory_type)
        return list(self.db.execute(stmt).all())
