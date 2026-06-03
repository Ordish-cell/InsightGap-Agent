from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.memory_repository import MemoryRepository


class MemoryService:
    def __init__(self):
        self._items: list[dict[str, Any]] = []

    def add_memory(self, user_id: int, content: str, memory_type: str = "working", importance: float = 0.0, metadata: dict[str, Any] | None = None, db: Session | None = None) -> dict[str, Any]:
        if db:
            item = MemoryRepository(db).create(user_id=user_id, content=content, memory_type=memory_type, importance=importance, metadata_json=metadata or {})
            return self._to_dict(item)
        item = {
            "id": len(self._items) + 1,
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata or {},
        }
        self._items.append(item)
        return item

    def search_memory(self, user_id: int, query: str = "", memory_type: str | None = None, min_importance: float = 0.0, db: Session | None = None) -> list[dict[str, Any]]:
        if db:
            return [self._to_dict(item) for item in MemoryRepository(db).search(user_id, query, memory_type, min_importance)]
        query_lower = query.lower()
        return [item for item in self._items if item["user_id"] == user_id and query_lower in item["content"].lower()]

    def summarize_memory(self, user_id: int, db: Session | None = None) -> dict[str, Any]:
        if db:
            repo = MemoryRepository(db)
            recent = repo.list_by_user(user_id)[:10]
            return {
                "counts": [{"memory_type": row[0], "count": row[1], "avg_importance": float(row[2] or 0)} for row in repo.counts_by_type(user_id)],
                "recent": [self._to_summary(item) for item in recent],
            }
        items = [item for item in self._items if item["user_id"] == user_id]
        return {"count": len(items), "summary": "; ".join(item["content"] for item in items[:5])}

    def consolidate_memory(self, user_id: int, db: Session | None = None) -> dict[str, Any]:
        if db:
            repo = MemoryRepository(db)
            promoted = 0
            for item in repo.search(user_id, memory_type="working", min_importance=0.7):
                repo.update(item, memory_type="episodic")
                promoted += 1
            for item in repo.search(user_id, memory_type="episodic", min_importance=0.8):
                repo.update(item, memory_type="semantic")
                promoted += 1
            return {"user_id": user_id, "promoted": promoted}
        return {"user_id": user_id, "promoted": 0, "mode": "mock"}

    def forget_memory(self, user_id: int, memory_id: int | None = None, db: Session | None = None) -> dict[str, Any]:
        if db:
            repo = MemoryRepository(db)
            if memory_id:
                item = repo.get_by_id(memory_id)
                if item and item.user_id == user_id:
                    db.delete(item)
                    db.commit()
                    return {"deleted": 1}
            return {"deleted": 0}
        before = len(self._items)
        self._items = [item for item in self._items if not (item["user_id"] == user_id and (memory_id is None or item["id"] == memory_id))]
        return {"deleted": before - len(self._items)}

    def _to_dict(self, item) -> dict[str, Any]:
        return {"id": item.id, "user_id": item.user_id, "content": item.content, "memory_type": item.memory_type, "importance": item.importance, "metadata": item.metadata_json or {}}

    def _to_summary(self, item) -> dict[str, Any]:
        content = "[masked sensitive memory]" if (item.metadata_json or {}).get("sensitive") else item.content
        return {"id": item.id, "memory_type": item.memory_type, "content": content, "importance": item.importance}


memory_service = MemoryService()
