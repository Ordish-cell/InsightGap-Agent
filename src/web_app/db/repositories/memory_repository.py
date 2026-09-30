from sqlalchemy import func, select, text, or_
import re

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Memory


class MemoryRepository(BaseRepository[Memory]):
    model = Memory

    def search_keywords(self, user_id, query, memory_types, min_importance=0, *, conversation_id=None, management=False):
        words = re.findall(r"[A-Za-z0-9_+.-]{2,}|[\u4e00-\u9fff]+", query)
        terms = []
        for word in words:
            if re.fullmatch(r"[\u4e00-\u9fff]+", word) and len(word) > 2:
                terms.extend(word[i:i + 2] for i in range(len(word) - 1))
            else:
                terms.append(word)
        terms = list(dict.fromkeys(terms))[:24]
        if not terms:
            return []
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.memory_type.in_(memory_types),
            Memory.importance >= min_importance,
            or_(*(Memory.content.ilike("%" + t.replace("_", "\\_") + "%", escape="\\") for t in terms)))
        if management:
            return list(self.db.scalars(stmt.order_by(Memory.importance.desc(), Memory.id.desc()).limit(128)))
        from datetime import UTC, datetime
        from src.web_app.memory.policy import PERSONAL_CATEGORIES
        status = Memory.metadata_json["status"].as_string()
        scopes = [Memory.scope == "user", (Memory.scope == "conversation") & (Memory.scope_id == conversation_id)]
        if management:
            scopes.append(Memory.scope == "legacy_unscoped")
        else:
            scopes.append((Memory.scope == "legacy_unscoped") & Memory.metadata_json["category"].as_string().in_(PERSONAL_CATEGORIES))
        stmt = stmt.where(or_(status.is_(None), status == "active"), or_(*scopes),
            Memory.metadata_json["visible_in_long_term_memory"].as_boolean().is_not(False),
            Memory.metadata_json["sensitive"].as_boolean().is_not(True),
            or_(Memory.expires_at.is_(None), Memory.expires_at > datetime.now(UTC)))
        return list(self.db.scalars(stmt.order_by(Memory.importance.desc(), Memory.id.desc()).limit(128)))

    def list_by_user(self, user_id: int) -> list[Memory]:
        return list(self.db.execute(select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())).scalars())

    def search(self, user_id: int, query: str = "", memory_type: str | None = None, min_importance: float = 0.0) -> list[Memory]:
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.importance >= min_importance,
            or_(Memory.metadata_json["status"].as_string().is_(None), Memory.metadata_json["status"].as_string() != "deleting")).order_by(Memory.importance.desc(), Memory.created_at.desc())
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
        """Management helper; query recall must never use these unrelated rows."""
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

    def list_long_term(self, user_id, memory_type=None, category=None, status=None, query=None, page=1, page_size=20, scope=None, scope_id=None) -> tuple:
        """Paginated visible long-term memories; active by default, status=all includes inactive."""
        # Defensive int cast — query params may arrive as strings
        page = max(1, int(page) if page else 1)
        page_size = max(1, min(100, int(page_size) if page_size else 20))
        types = ["semantic", "episodic"]
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.memory_type.in_(types)).order_by(Memory.updated_at.desc(), Memory.importance.desc())
        if memory_type and memory_type in types: stmt = stmt.where(Memory.memory_type == memory_type)
        if scope: stmt = stmt.where(Memory.scope == scope)
        if scope_id: stmt = stmt.where(Memory.scope_id == scope_id)
        if query: stmt = stmt.where(Memory.content.ilike(f"%{query}%"))
        rows = list(self.db.execute(stmt).scalars())
        explicit_status = status or "active"
        filtered = []
        for m in rows:
            meta = m.metadata_json or {}
            if not meta.get("visible_in_long_term_memory", True) and explicit_status != "all": continue
            mem_status = meta.get("status", "active")
            if mem_status == "deleting": continue
            if explicit_status != "all" and mem_status != explicit_status: continue
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
        from src.web_app.services.memory_service import memory_service
        rows = list(self.db.scalars(stmt))
        rows = [m for m in rows if memory_service.is_recallable(m, self.db, m.scope_id)]
        return rows[offset:offset + limit] if limit is not None else rows[offset:]

    def count_for_vector_backfill(
        self,
        user_id: int | None = None,
        memory_types: list[str] | None = None,
        include_working: bool = False,
    ) -> int:
        """Count memories eligible for Qdrant vector indexing."""
        return len(self.list_for_vector_backfill(user_id=user_id, memory_types=memory_types, include_working=include_working))
