import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.memory.extractor import memory_extractor


class MemoryService:
    def __init__(self):
        self._items: list[dict[str, Any]] = []

    def add_memory(
        self,
        user_id: int,
        content: str,
        memory_type: str = "working",
        importance: float = 0.0,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> dict[str, Any]:
        if db:
            item = MemoryRepository(db).create(
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata_json=metadata or {},
            )
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

    def add_with_dedup(
        self,
        user_id: int,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.0,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> dict[str, Any] | None:
        if db:
            existing = self._find_similar(user_id, content, memory_type, db)
            if existing:
                return self._update_existing(existing, importance, metadata, db)
        return self.add_memory(user_id, content, memory_type, importance, metadata, db)

    def _find_similar(self, user_id: int, content: str, memory_type: str, db: Session) -> Any:
        existing = MemoryRepository(db).search_by_type(user_id, memory_type=memory_type, min_importance=0.3)
        for mem in existing:
            if self._similarity(content, mem.content) >= 0.55:
                return mem
        return None

    def _similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        t1 = re.sub(r"[^\w一-鿿]", "", text1.lower())
        t2 = re.sub(r"[^\w一-鿿]", "", text2.lower())
        if not t1 or not t2:
            return 0.0
        if t1 in t2 or t2 in t1:
            return 0.85
        chars1 = set(t1)
        chars2 = set(t2)
        if not chars1 or not chars2:
            return 0.0
        intersection = chars1 & chars2
        union = chars1 | chars2
        jaccard = len(intersection) / len(union)
        words1 = set(self._ngrams(t1, 3))
        words2 = set(self._ngrams(t2, 3))
        if not words1 or not words2:
            return jaccard
        word_intersection = words1 & words2
        word_union = words1 | words2
        word_jaccard = len(word_intersection) / len(word_union)
        return 0.4 * jaccard + 0.6 * word_jaccard

    def _ngrams(self, text: str, n: int) -> list[str]:
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def _update_existing(self, existing: Any, importance: float, metadata: dict[str, Any] | None, db: Session) -> dict[str, Any]:
        repo = MemoryRepository(db)
        current_meta = dict(existing.metadata_json or {})
        evidence_count = current_meta.get("evidence_count", 1) + 1
        updated_importance = max(existing.importance, importance)
        updated_meta = {
            **current_meta,
            **(metadata or {}),
            "last_seen_at": datetime.now(UTC).isoformat(),
            "evidence_count": evidence_count,
            "updated_from": existing.importance,
        }
        # Boost importance slightly with repeated evidence
        if evidence_count >= 3:
            updated_importance = min(0.98, updated_importance + 0.05)
        repo.update(
            existing,
            importance=updated_importance,
            metadata_json=updated_meta,
        )
        return self._to_dict(existing)

    def search_memory(
        self,
        user_id: int,
        query: str = "",
        memory_type: str | None = None,
        min_importance: float = 0.0,
        db: Session | None = None,
    ) -> list[dict[str, Any]]:
        if db:
            return [self._to_dict(item) for item in MemoryRepository(db).search(user_id, query, memory_type, min_importance)]
        query_lower = query.lower()
        return [item for item in self._items if item["user_id"] == user_id and query_lower in item["content"].lower()]

    def get_semantic_memories(self, user_id: int, db: Session, min_importance: float = 0.3) -> list[dict[str, Any]]:
        return [self._to_dict(item) for item in MemoryRepository(db).search(user_id, memory_type="semantic", min_importance=min_importance)]

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

    def extract_and_save(
        self,
        user_id: int,
        user_input: str,
        agent_output: str = "",
        page_context: dict[str, Any] | None = None,
        feed_card_context: dict[str, Any] | None = None,
        matched_skill: dict[str, Any] | None = None,
        created_skill_draft: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> dict[str, Any]:
        extraction = memory_extractor.extract(
            user_input=user_input,
            agent_output=agent_output,
            page_context=page_context,
            feed_card_context=feed_card_context,
            matched_skill=matched_skill,
            created_skill_draft=created_skill_draft,
        )
        saved: dict[str, list[dict[str, Any]]] = {"working": [], "episodic": [], "semantic": []}

        for mem in extraction.get("working_memories", []):
            result = self.add_memory(
                user_id, mem["content"], memory_type="working",
                importance=mem.get("importance", 0.3),
                metadata={"category": mem.get("category", ""), "source": mem.get("source", "")},
                db=db,
            )
            saved["working"].append(result)

        for mem in extraction.get("episodic_memories", []):
            result = self.add_memory(
                user_id, mem["content"], memory_type="episodic",
                importance=mem.get("importance", 0.5),
                metadata={"category": mem.get("category", ""), "source": mem.get("source", "")},
                db=db,
            )
            saved["episodic"].append(result)

        for mem in extraction.get("semantic_memories", []):
            result = self.add_with_dedup(
                user_id, mem["content"], memory_type="semantic",
                importance=mem.get("importance", 0.8),
                metadata={
                    "category": mem.get("category", ""),
                    "source": mem.get("source", "home_chat"),
                    "confidence": mem.get("confidence", 0.8),
                    "evidence_count": 1,
                    "last_seen_at": datetime.now(UTC).isoformat(),
                },
                db=db,
            )
            if result:
                saved["semantic"].append(result)

        if extraction.get("should_consolidate") and db:
            self.consolidate_memory(user_id, db)

        return {"extraction": extraction, "saved": saved}

    def _to_dict(self, item) -> dict[str, Any]:
        return {
            "id": item.id,
            "user_id": item.user_id,
            "content": item.content,
            "memory_type": item.memory_type,
            "importance": item.importance,
            "metadata": item.metadata_json or {},
        }

    def _to_summary(self, item) -> dict[str, Any]:
        content = "[masked sensitive memory]" if (item.metadata_json or {}).get("sensitive") else item.content
        return {"id": item.id, "memory_type": item.memory_type, "content": content, "importance": item.importance}


memory_service = MemoryService()
