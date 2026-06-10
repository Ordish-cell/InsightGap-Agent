from sqlalchemy import func, select, text

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

    def get_by_ids(self, user_id: int, ids: list[int]) -> list[Memory]:
        """Fetch memories by ID list — used after Qdrant returns memory_id hits."""
        if not ids:
            return []
        stmt = (
            select(Memory)
            .where(Memory.user_id == user_id, Memory.id.in_(ids))
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
        )
        return list(self.db.execute(stmt).scalars())

    def list_recent_important(
        self,
        user_id: int,
        memory_type: str = "semantic",
        min_importance: float = 0.8,
        limit: int = 5,
    ) -> list[Memory]:
        """Fallback when Qdrant and ILIKE both yield nothing."""
        stmt = (
            select(Memory)
            .where(
                Memory.user_id == user_id,
                Memory.memory_type == memory_type,
                Memory.importance >= min_importance,
            )
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars())

    def list_long_term(self, user_id, memory_type=None, category=None, status=None, query=None, page=1, page_size=20) -> tuple:
        """Paginated long-term memories (semantic+episodic, visible, not superseded by default)."""
        # Defensive int cast — query params may arrive as strings
        page = max(1, int(page) if page else 1)
        page_size = max(1, min(100, int(page_size) if page_size else 20))
        types = ["semantic", "episodic"]
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.memory_type.in_(types)).order_by(Memory.updated_at.desc(), Memory.importance.desc())
        if memory_type and memory_type in types: stmt = stmt.where(Memory.memory_type == memory_type)
        if query: stmt = stmt.where(Memory.content.ilike(f"%{query}%"))
        rows = list(self.db.execute(stmt).scalars())
        show_superseded = status == "superseded"
        explicit_status = status
        filtered = []
        for m in rows:
            meta = m.metadata_json or {}
            if not meta.get("visible_in_long_term_memory", False): continue
            mem_status = meta.get("status", "active")
            if mem_status == "superseded" and not show_superseded: continue
            if not explicit_status and mem_status != "active": continue
            if explicit_status and mem_status != explicit_status: continue
            if category and meta.get("category") != category: continue
            filtered.append(m)
        total = len(filtered)
        offset = (page - 1) * page_size
        return filtered[offset:offset + page_size], total

    def list_for_vector_backfill(
        self,
        user_id: int | None = None,
        memory_types: list[str] | None = None,
        include_working: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories eligible for Qdrant vector indexing.

        Default: semantic + episodic only, non-empty content,
        ordered by created_at ASC for stable backfill progress.
        """
        types = memory_types or ["semantic", "episodic"]
        stmt = (
            select(Memory)
            .where(
                Memory.memory_type.in_(types),
                Memory.content.isnot(None),
                Memory.content != "",
            )
            .order_by(Memory.created_at.asc(), Memory.id.asc())
        )
        if user_id is not None:
            stmt = stmt.where(Memory.user_id == user_id)
        if not include_working:
            stmt = stmt.where(Memory.memory_type != "working")
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.db.execute(stmt).scalars())

    def count_for_vector_backfill(
        self,
        user_id: int | None = None,
        memory_types: list[str] | None = None,
        include_working: bool = False,
    ) -> int:
        """Count memories eligible for Qdrant vector indexing."""
        types = memory_types or ["semantic", "episodic"]
        stmt = (
            select(func.count(Memory.id))
            .where(
                Memory.memory_type.in_(types),
                Memory.content.isnot(None),
                Memory.content != "",
            )
        )
        if user_id is not None:
            stmt = stmt.where(Memory.user_id == user_id)
        if not include_working:
            stmt = stmt.where(Memory.memory_type != "working")
        return int(self.db.execute(stmt).scalar() or 0)
