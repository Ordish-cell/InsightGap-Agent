import logging
import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.memory.extractor import memory_extractor

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(self):
        self._items: list[dict[str, Any]] = []
        self._qdrant_store = None
        self._qdrant_init_attempted = False
        self._last_search_backend = "not_searched"
        self._last_qdrant_hits = 0

    def _get_qdrant_store(self):
        """Lazy-init QdrantMemoryStore. Returns None if Qdrant is unavailable."""
        if self._qdrant_store is not None:
            return self._qdrant_store
        if self._qdrant_init_attempted:
            return None
        self._qdrant_init_attempted = True
        try:
            from src.web_app.core.config import settings
            from src.web_app.memory.qdrant_memory_store import QdrantMemoryStore
            if not settings.qdrant_url:
                logger.info("memory.qdrant_skipped: QDRANT_URL not configured")
                return None
            store = QdrantMemoryStore()
            store.ensure_collection()
            self._qdrant_store = store
            logger.info("memory.qdrant_store_ready", extra={"collection": store.collection})
            return store
        except Exception:
            logger.warning("memory.qdrant_init_failed", exc_info=True)
            return None

    def add_memory(
        self,
        user_id: int,
        content: str,
        memory_type: str = "working",
        importance: float = 0.0,
        source_type: str = "",
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> dict[str, Any]:
        if db:
            item = MemoryRepository(db).create(
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                source_type=source_type,
                metadata_json=metadata or {},
            )
            # ── Fire-and-forget Qdrant upsert ─────────────────────────
            store = self._get_qdrant_store()
            if store is not None:
                try:
                    point_id = store.upsert_memory(
                        memory_id=item.id,
                        user_id=user_id,
                        content=content,
                        memory_type=memory_type,
                        importance=importance,
                        source_type=source_type,
                        metadata=metadata,
                    )
                    # Mark as indexed in PG metadata (best-effort)
                    try:
                        MemoryRepository(db).update(
                            item,
                            qdrant_point_id=point_id,
                            metadata_json={
                                **(item.metadata_json or {}),
                                "qdrant_indexed": True,
                            },
                        )
                    except Exception:
                        pass  # metadata update is non-critical
                except Exception:
                    logger.warning("memory.qdrant_upsert_failed", exc_info=True)
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
        source_type: str = "",
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> dict[str, Any] | None:
        if db:
            existing = self._find_similar(user_id, content, memory_type, db)
            if existing:
                return self._update_existing(existing, importance, metadata, db)
        return self.add_memory(user_id, content, memory_type, importance, source_type, metadata, db)

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
        store = self._get_qdrant_store()
        if store is not None and existing.qdrant_point_id:
            try:
                store.upsert_memory(memory_id=existing.id, user_id=existing.user_id, content=existing.content, memory_type=existing.memory_type, importance=updated_importance, source_type=existing.source_type or "", metadata=updated_meta, point_id=existing.qdrant_point_id)
            except Exception: pass
        return self._to_dict(existing)

    def search_memory(
        self,
        user_id: int,
        query: str = "",
        memory_type: str | None = None,
        min_importance: float = 0.0,
        db: Session | None = None,
        *,
        memory_types: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Search long-term memory with Qdrant → ILIKE → recent-important fallback."""
        from datetime import UTC, datetime as _dt
        from src.web_app.services.user_growth_service import user_growth_service as _ugs

        if not db:
            query_lower = query.lower()
            return [item for item in self._items if item["user_id"] == user_id and query_lower in item["content"].lower()]

        # ── Helper closures ─────────────────────────────────────────────
        def _status_allowed(m) -> bool:
            if hasattr(m, "metadata_json"):
                s = (m.metadata_json or {}).get("status", "active")
            else:
                s = (m.get("metadata", {}) or {}).get("status", "active")
            return s not in ("superseded", "archived")

        def _effective_importance(mem_dict: dict) -> float:
            imp = float(mem_dict.get("importance", 0) or 0)
            meta = mem_dict.get("metadata", {}) or {}
            status = meta.get("status", "active")
            if status == "superseded":
                imp *= 0.50
            elif status == "archived":
                imp *= 0.25
            if meta.get("stability") == "temporary":
                imp *= 0.80
            return imp

        def _recency_score(meta: dict) -> float:
            now = _dt.now(UTC)
            ts_str = meta.get("last_seen_at") or meta.get("updated_at") or meta.get("created_at") or ""
            if not ts_str:
                return 0.0
            try:
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                ts = _dt.fromisoformat(ts_str)
                age_days = (now - ts).total_seconds() / 86400.0
                # Half-life of 30 days
                return max(0.0, 1.0 / (1.0 + age_days / 30.0))
            except Exception:
                return 0.0

        # Determine which memory types to search
        types_for_search: list[str] | None = None
        if memory_types:
            types_for_search = memory_types
        elif memory_type:
            types_for_search = [memory_type]

        results: list[dict[str, Any]] = []
        repo = MemoryRepository(db)

        # ── Tier 1: Qdrant semantic search ────────────────────────────
        if query:
            store = self._get_qdrant_store()
            if store is not None:
                try:
                    qdrant_hits = store.search_memory(
                        user_id=user_id,
                        query=query,
                        memory_types=types_for_search,
                        limit=limit,
                        score_threshold=0.25,
                    )
                    self._last_search_backend = "qdrant"
                    self._last_qdrant_hits = len(qdrant_hits)

                    if qdrant_hits:
                        memory_ids = [int(h["memory_id"]) for h in qdrant_hits]
                        memories = repo.get_by_ids(user_id=user_id, ids=memory_ids)
                        # Filter by min_importance, memory_type, and status
                        if min_importance > 0:
                            memories = [m for m in memories if m.importance >= min_importance]
                        if types_for_search:
                            memories = [m for m in memories if m.memory_type in types_for_search]
                        memories = [m for m in memories if _status_allowed(m)]
                        # Sort: 65% Qdrant + 25% effective importance + 10% recency
                        score_map = {int(h["memory_id"]): h["score"] for h in qdrant_hits}
                        memories.sort(
                            key=lambda m: (
                                0.65 * score_map.get(m.id, 0)
                                + 0.25 * _effective_importance(self._to_dict(m))
                                + 0.10 * _recency_score(m.metadata_json or {})
                            ),
                            reverse=True,
                        )
                        results = []
                        for m in memories[:limit]:
                            d = self._to_dict(m)
                            d["_qdrant_score"] = score_map.get(m.id, 0)
                            meta = m.metadata_json or {}
                            if meta.get("confidence", 0.95) < 0.55:
                                d["_low_confidence"] = True
                            results.append(d)
                        return results
                except Exception:
                    logger.warning("memory.qdrant_search_failed", exc_info=True)
                    self._last_search_backend = "qdrant_search_failed"
                    self._last_qdrant_hits = 0

        # ── Tier 2: PostgreSQL ILIKE fallback ──────────────────────────
        pg_memories = repo.search(
            user_id, query=query if query else "",
            memory_type=memory_type,
            min_importance=min_importance,
        )
        if pg_memories:
            self._last_search_backend = "postgres_like"
            self._last_qdrant_hits = 0
            pg_memories = [m for m in pg_memories if _status_allowed(m)]
            return [self._to_dict(m) for m in pg_memories[:limit]]

        # ── Tier 3: recent important semantic memories ─────────────────
        fallback_memories = repo.list_recent_important(
            user_id=user_id,
            memory_type="semantic",
            min_importance=0.7,
            limit=limit,
        )
        if fallback_memories:
            self._last_search_backend = "recent_important_fallback"
            self._last_qdrant_hits = 0
            fallback_memories = [m for m in fallback_memories if _status_allowed(m)]
            return [self._to_dict(m) for m in fallback_memories]

        self._last_search_backend = "no_results"
        self._last_qdrant_hits = 0
        return []

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

    _SEMANTIC_CATEGORIES = {"preference", "negative_preference", "project_goal", "tech_stack", "boundary", "answer_preference", "name_preference", "language_preference", "tone_preference", "workflow_pattern"}

    def consolidate_memory(self, user_id: int, db: Session | None = None) -> dict[str, Any]:
        if not db: return {"user_id": user_id, "promoted": 0, "mode": "mock"}
        repo = MemoryRepository(db)
        promoted_w_to_e = 0; promoted_e_to_s = 0
        now_ts = datetime.now(UTC).isoformat()
        # working→episodic: importance>=0.7
        for item in repo.search(user_id, memory_type="working", min_importance=0.7):
            updated_meta = dict(item.metadata_json or {})
            updated_meta.update({"visible_in_long_term_memory": True, "status": "active", "stability": "medium_term", "consolidated_at": now_ts, "consolidated_from": "working"})
            repo.update(item, memory_type="episodic", metadata_json=updated_meta)
            promoted_w_to_e += 1
        # episodic→semantic: importance>=0.8 AND stability!="temporary" AND evidence_count>=2 AND category in _SEMANTIC_CATEGORIES
        for item in repo.search(user_id, memory_type="episodic", min_importance=0.8):
            meta = item.metadata_json or {}
            if meta.get("stability") == "temporary" or meta.get("evidence_count", 1) < 2: continue
            if meta.get("category", "") not in self._SEMANTIC_CATEGORIES: continue
            updated_meta = dict(meta)
            updated_meta.update({"stability": "long_term", "consolidated_at": now_ts, "consolidated_from": "episodic"})
            repo.update(item, memory_type="semantic", metadata_json=updated_meta)
            promoted_e_to_s += 1
        return {"user_id": user_id, "promoted_working_to_episodic": promoted_w_to_e, "promoted_episodic_to_semantic": promoted_e_to_s, "total_promoted": promoted_w_to_e + promoted_e_to_s}

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

    @staticmethod
    def _is_protected_memory(memory: Any) -> bool:
        """Returns True if memory is protected and should not be archived by forgetting strategies."""
        meta = memory.metadata_json if hasattr(memory, "metadata_json") else memory.get("metadata", {}) if isinstance(memory, dict) else {}
        status = meta.get("status", "active") if meta else "active"
        is_protected = meta.get("protected", False) if meta else False
        importance = memory.importance if hasattr(memory, "importance") else memory.get("importance", 0)
        return status == "superseded" or is_protected or (status == "active" and importance >= 0.8)

    def _archive_memory(self, memory: Any, reason: str, db: Session) -> dict[str, Any]:
        repo = MemoryRepository(db)
        now_ts = datetime.now(UTC).isoformat()
        current_meta = dict(memory.metadata_json if hasattr(memory, "metadata_json") else memory.get("metadata", {}))
        current_meta["status"] = "archived"
        current_meta["archived_at"] = now_ts
        current_meta["archive_reason"] = reason
        repo.update(memory, metadata_json=current_meta)
        return self._to_dict(memory)

    def forget_by_importance(self, user_id: int, threshold: float = 0.2, memory_type: str | None = None, db: Session | None = None) -> dict[str, Any]:
        """Archive memories with importance below threshold. Skips protected memories."""
        if not db: return {"archived": 0, "skipped_protected": 0, "strategy": "importance", "details": []}
        repo = MemoryRepository(db)
        archived = 0; skipped = 0; details: list[dict[str, Any]] = []
        types_to_check = [memory_type] if memory_type else ["semantic", "episodic"]
        for mtype in types_to_check:
            for item in repo.search(user_id, memory_type=mtype, min_importance=0.0):
                if item.importance >= threshold: continue
                if self._is_protected_memory(item):
                    skipped += 1; continue
                self._archive_memory(item, f"forget_by_importance: importance {item.importance:.2f} < {threshold}", db)
                archived += 1
                details.append({"id": item.id, "content": item.content[:80], "importance": item.importance, "type": mtype})
        return {"archived": archived, "skipped_protected": skipped, "strategy": "importance", "threshold": threshold, "details": details}

    def forget_by_time(self, user_id: int, max_age_days: int = 90, memory_type: str | None = None, db: Session | None = None) -> dict[str, Any]:
        """Archive memories not seen within max_age_days. Skips protected memories."""
        if not db: return {"archived": 0, "skipped_protected": 0, "strategy": "time", "details": []}
        repo = MemoryRepository(db)
        archived = 0; skipped = 0; details: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        types_to_check = [memory_type] if memory_type else ["semantic", "episodic"]
        for mtype in types_to_check:
            for item in repo.search(user_id, memory_type=mtype, min_importance=0.0):
                meta = item.metadata_json or {}
                last_ts_str = meta.get("last_seen_at") or meta.get("updated_at") or str(item.created_at) if hasattr(item, "created_at") else ""
                try:
                    if last_ts_str and last_ts_str.endswith("Z"):
                        last_ts_str = last_ts_str[:-1] + "+00:00"
                    last_ts = datetime.fromisoformat(str(last_ts_str)) if last_ts_str else None
                except Exception:
                    last_ts = None
                if last_ts is None:
                    continue
                age_days = (now - last_ts.replace(tzinfo=UTC)).total_seconds() / 86400.0 if last_ts.tzinfo else (now - last_ts).total_seconds() / 86400.0
                if age_days <= max_age_days: continue
                if self._is_protected_memory(item):
                    skipped += 1; continue
                self._archive_memory(item, f"forget_by_time: last seen {age_days:.0f} days ago > {max_age_days}", db)
                archived += 1
                details.append({"id": item.id, "content": item.content[:80], "age_days": round(age_days, 1), "type": mtype})
        return {"archived": archived, "skipped_protected": skipped, "strategy": "time", "max_age_days": max_age_days, "details": details}

    def forget_by_capacity(self, user_id: int, memory_type: str = "semantic", max_capacity: int = 500, db: Session | None = None) -> dict[str, Any]:
        """Archive lowest effective_importance memories when count exceeds max_capacity. Skips protected."""
        if not db: return {"archived": 0, "skipped_protected": 0, "strategy": "capacity", "details": []}
        repo = MemoryRepository(db)
        all_items = repo.search(user_id, memory_type=memory_type, min_importance=0.0)
        if len(all_items) <= max_capacity:
            return {"archived": 0, "skipped_protected": 0, "strategy": "capacity", "count": len(all_items), "max_capacity": max_capacity, "details": []}
        # Sort by effective importance (lowest first), protected go last
        def eff_imp(item):
            return float(item.importance or 0) * (0.25 if (item.metadata_json or {}).get("status") == "archived" else 0.5 if (item.metadata_json or {}).get("status") == "superseded" else 1.0)
        sorted_items = sorted(all_items, key=lambda x: (1 if self._is_protected_memory(x) else 0, eff_imp(x)))
        to_remove = len(all_items) - max_capacity
        archived = 0; skipped = 0; details: list[dict[str, Any]] = []
        for item in sorted_items:
            if archived >= to_remove: break
            if self._is_protected_memory(item):
                skipped += 1; continue
            self._archive_memory(item, f"forget_by_capacity: exceeded max {max_capacity}", db)
            archived += 1
            details.append({"id": item.id, "content": item.content[:80], "importance": item.importance, "effective_importance": eff_imp(item)})
        return {"archived": archived, "skipped_protected": skipped, "strategy": "capacity", "max_capacity": max_capacity, "original_count": len(all_items), "details": details}

    def extract_and_save(self, user_id, user_input, agent_output="", page_context=None, feed_card_context=None, matched_skill=None, created_skill_draft=None, db=None, run_id="") -> dict:
        """Sync extraction using regex only."""
        extraction = memory_extractor.extract(
            user_input=user_input, agent_output=agent_output, page_context=page_context,
            feed_card_context=feed_card_context, matched_skill=matched_skill, created_skill_draft=created_skill_draft)
        return self._save_extracted(user_id, extraction, db, run_id)

    async def async_extract_and_save(self, user_id, user_input, agent_output="", page_context=None, feed_card_context=None, matched_skill=None, created_skill_draft=None, db=None, run_id="", use_llm=True, thread_id="") -> dict:
        """Async extraction with LLM primary + regex fallback."""
        extraction = None
        llm_used = False
        if use_llm and db:
            try:
                from src.web_app.memory.extractor import LlmMemoryExtractor
                llm_extractor = LlmMemoryExtractor()
                extraction = await llm_extractor.extract(
                    db=db, run_id=run_id, thread_id=thread_id, user_id=user_id,
                    user_input=user_input, agent_output=agent_output,
                    page_context=page_context, feed_card_context=feed_card_context,
                    matched_skill=matched_skill, created_skill_draft=created_skill_draft)
                llm_used = True
            except Exception:
                logger.warning("memory.async_extract_and_save: LLM failed, using regex fallback", exc_info=True)
        if extraction is None:
            extraction = memory_extractor.extract(
                user_input=user_input, agent_output=agent_output, page_context=page_context,
                feed_card_context=feed_card_context, matched_skill=matched_skill, created_skill_draft=created_skill_draft)
        result = self._save_extracted(user_id, extraction, db, run_id)
        result["llm_used"] = llm_used
        return result

    def _save_extracted(self, user_id, extraction, db, run_id) -> dict:
        saved = {"working": [], "episodic": [], "semantic": []}
        filtered_out = {"episodic": 0, "semantic": 0}
        now_ts = datetime.now(UTC).isoformat()

        # working: low-barrier, visible_in_long_term_memory=False
        for mem in extraction.get("working_memories", []):
            result = self.add_memory(user_id, mem["content"], memory_type="working",
                importance=mem.get("importance", 0.3),
                metadata={"category": mem.get("category", ""), "source": mem.get("source", ""),
                          "visible_in_long_term_memory": False, "stability": "temporary",
                          "status": "active", "confidence": mem.get("confidence", 0.95)}, db=db)
            saved["working"].append(result)

        # episodic: importance>=0.65 AND confidence>=0.65
        for mem in extraction.get("episodic_memories", []):
            importance = mem.get("importance", 0.5)
            confidence = mem.get("confidence", 0.80)
            if importance < 0.65 or confidence < 0.65:
                filtered_out["episodic"] += 1; continue
            result = self.add_with_dedup(user_id, mem["content"], memory_type="episodic",
                importance=importance,
                metadata={"category": mem.get("category", ""), "source": mem.get("source", ""),
                          "visible_in_long_term_memory": True, "stability": mem.get("stability", "medium_term"),
                          "status": "active", "evidence_count": 1, "last_seen_at": now_ts, "confidence": confidence}, db=db)
            if result: saved["episodic"].append(result)

        # semantic: importance>=0.70 AND confidence>=0.65
        for mem in extraction.get("semantic_memories", []):
            importance = mem.get("importance", 0.8)
            confidence = mem.get("confidence", 0.80)
            if importance < 0.70 or confidence < 0.65:
                filtered_out["semantic"] += 1; continue
            result = self.add_with_dedup(user_id, mem["content"], memory_type="semantic",
                importance=importance,
                metadata={"category": mem.get("category", ""), "source": mem.get("source", "home_chat"),
                          "visible_in_long_term_memory": True, "stability": mem.get("stability", "long_term"),
                          "status": "active", "evidence_count": 1, "last_seen_at": now_ts, "confidence": confidence}, db=db)
            if result: saved["semantic"].append(result)

        # Auto consolidate + reflect triggers
        if db:
            should_consolidate = extraction.get("should_consolidate", False)
            saved_sem_count = len(saved["semantic"])
            has_high_episodic = any(m.get("importance", 0) >= 0.75 for m in extraction.get("episodic_memories", []) if m.get("importance", 0) >= 0.65)
            if saved_sem_count > 0 or has_high_episodic: should_consolidate = True
            if should_consolidate:
                try:
                    self.consolidate_memory(user_id, db)
                except Exception:
                    logger.warning("memory.auto_consolidate_failed", exc_info=True)
            try:
                from collections import Counter
                repo = MemoryRepository(db)
                all_sem, _ = repo.list_long_term(user_id, memory_type="semantic", page=1, page_size=500)
                cat_counts = Counter()
                for m in all_sem:
                    if (m.metadata_json or {}).get("status", "active") == "active":
                        cat = (m.metadata_json or {}).get("category", "")
                        if cat: cat_counts[cat] += 1
                if any(c >= 4 for c in cat_counts.values()):
                    from src.web_app.services.user_growth_service import user_growth_service as _ugs
                    _ugs.reflect_user_profile(user_id, db)
            except Exception:
                logger.warning("memory.auto_reflect_failed", exc_info=True)

        total_saved = len(saved["working"]) + len(saved["episodic"]) + len(saved["semantic"])
        logger.info("memory.extract_and_save: run_id=%s working=%d episodic=%d semantic=%d filtered_episodic=%d filtered_semantic=%d",
                    run_id, len(saved["working"]), len(saved["episodic"]), len(saved["semantic"]),
                    filtered_out["episodic"], filtered_out["semantic"])
        return {"extraction": extraction, "saved": saved, "filtered_out": filtered_out, "total_saved": total_saved}

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
