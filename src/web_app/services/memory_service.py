import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.memory.extractor import memory_extractor
from src.web_app.memory.policy import eligible, infer_scope, runtime_policy, validate_source
from src.web_app.memory.facts import compare, evidence, fact_key, normalized, source_order
from src.web_app.services.deletion_guard import guarded_transition

logger = logging.getLogger(__name__)


class MemoryService:
    _QDRANT_MEMORY_TYPES = {"semantic", "episodic"}

    def __init__(self):
        self._items: list[dict[str, Any]] = []
        self._qdrant_store = None
        self._qdrant_init_attempted = False
        self._qdrant_retry_after = 0.0
        self._last_search_backend = "not_searched"
        self._last_qdrant_hits = 0

    def _get_qdrant_store(self):
        """Lazy-init QdrantMemoryStore. Returns None if Qdrant is unavailable."""
        if self._qdrant_store is not None:
            return self._qdrant_store
        now = time.monotonic()
        if self._qdrant_init_attempted and now < self._qdrant_retry_after:
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
            self._qdrant_retry_after = now + 60.0
            return None

    def _delete_memory_vector(self, memory_id: int | str) -> str | None:
        """Delete only a memory vector from the memory collection."""
        store = self._get_qdrant_store()
        if store is None:
            warning = "Qdrant memory store is not configured or unavailable"
            logger.warning("memory.vector_cleanup_skipped memory_id=%s reason=%s", memory_id, warning)
            return warning
        try:
            store.delete_by_memory_id(memory_id)
            return None
        except Exception as exc:
            warning = f"Memory vector cleanup failed: {exc}"
            logger.warning("memory.vector_cleanup_failed memory_id=%s error=%s", memory_id, exc, exc_info=True)
            return warning

    @guarded_transition
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
        # Backward-compatible positional form used by older callers:
        # add_memory(user_id, content, type, importance, metadata, db)
        if db is None and isinstance(metadata, Session):
            db = metadata
            metadata = source_type if isinstance(source_type, dict) else {}
            source_type = ""
        elif isinstance(source_type, dict) and metadata is None:
            metadata = source_type
            source_type = ""

        metadata = dict(metadata or {})
        if db and runtime_policy(db).get("writes_blocked"):
            raise ValueError("memory_writes_disabled")
        metadata.setdefault("status", "active")
        metadata.setdefault("visible_in_long_term_memory", memory_type in self._QDRANT_MEMORY_TYPES)
        if memory_type == "working":
            metadata["visible_in_long_term_memory"] = False

        if db:
            from src.web_app.services.deletion_guard import check_conversation
            from src.web_app.agent.runtime.chat_control import execution
            from src.web_app.models.orm import AgentRun
            metadata = validate_source(db, user_id, metadata)
            token = execution.get()
            source_run = metadata.get("run_id") or metadata.get("agent_run_id") or (token.run_id if token else None)
            if source_run and str(source_run).isdigit():
                owner_run = db.get(AgentRun, int(source_run))
                if owner_run and owner_run.user_id == user_id:
                    metadata.update(run_id=owner_run.id, conversation_id=owner_run.conversation_id)
            check_conversation(db, user_id, metadata.get("conversation_id"), require_exists=bool(metadata.get("conversation_id")))
            scope, scope_id = infer_scope(metadata, metadata.get("conversation_id", ""))
            metadata["fact_key"] = fact_key(content, metadata)
            metadata["evidence"] = evidence(metadata)
            metadata["evidence_count"] = len({e["message_id"] for e in metadata["evidence"]})
            item = MemoryRepository(db).create(
                scope=scope, scope_id=scope_id,
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                source_type=source_type,
                metadata_json=metadata or {},
            )
            return self._index_saved(item, db)
        item = {
            "id": len(self._items) + 1,
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata or {},
            "scope": infer_scope(metadata, metadata.get("conversation_id", ""))[0],
            "scope_id": infer_scope(metadata, metadata.get("conversation_id", ""))[1],
            "ok": True,
            "qdrant_point_id": None,
            "qdrant_indexed": False,
            "deduped": False,
            "updated_existing": False,
            "error": None,
            "category": (metadata or {}).get("category", ""),
            "status": (metadata or {}).get("status", "active"),
        }
        self._items.append(item)
        return item

    def _index_saved(self, item, db):
        user_id, content, memory_type = item.user_id, item.content, item.memory_type
        importance, source_type, metadata = item.importance, item.source_type, item.metadata_json
        result = self._to_dict(item)
        qdrant_point_id = None
        qdrant_indexed = False
        qdrant_error = None
        # PostgreSQL is authoritative; vector indexing is best-effort.
        store = self._get_qdrant_store() if memory_type in self._QDRANT_MEMORY_TYPES and eligible(result, item.scope_id) else None
        if store is not None:
            try:
                from uuid import NAMESPACE_URL, uuid5
                qdrant_point_id = store.upsert_memory(
                    memory_id=item.id,
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    source_type=source_type,
                    metadata={**metadata, "scope": item.scope, "scope_id": item.scope_id},
                    point_id=item.qdrant_point_id or str(uuid5(NAMESPACE_URL, f"memory:{user_id}:{item.id}")),
                )
                qdrant_indexed = True
                # Mark as indexed in PG metadata (best-effort)
                try:
                    MemoryRepository(db).update(
                        item,
                        qdrant_point_id=qdrant_point_id,
                        metadata_json={
                            **(item.metadata_json or {}),
                            "qdrant_indexed": True,
                        },
                    )
                    result["metadata"] = {**metadata, "qdrant_indexed": True}
                except Exception:
                    db.rollback()  # The memory transaction already committed.
            except Exception as exc:
                qdrant_error = str(exc)[:200]
                logger.warning("memory.qdrant_upsert_failed", exc_info=True)
        result.update({
            "ok": True,
            "qdrant_point_id": qdrant_point_id,
            "qdrant_indexed": qdrant_indexed,
            "deduped": False,
            "updated_existing": False,
            "error": qdrant_error,
            "category": (metadata or {}).get("category", ""),
            "status": (metadata or {}).get("status", "active"),
        })
        graph_result = self._sync_memory_graph(user_id, item) if eligible(result, item.scope_id) else {"synced": False}
        if graph_result.get("warning"):
            result["graph_warning"] = graph_result["warning"]
        result["graph_indexed"] = bool(graph_result.get("synced"))
        # Durable retry markers live beside the authoritative fact. Index failures
        # never roll back the fact or its version transition.
        from src.web_app.core.config import settings
        pending = bool(eligible(result, item.scope_id) and (
            (settings.qdrant_url and not qdrant_indexed) or
            (settings.enable_neo4j and settings.neo4j_memory_graph_enabled and not graph_result.get("synced"))))
        try:
            MemoryRepository(db).update(item, metadata_json={**(item.metadata_json or {}),
                "index_pending": pending, "graph_indexed": result["graph_indexed"]})
        except Exception:
            db.rollback()
            logger.warning("memory.index_retry_marker_failed memory_id=%s", item.id, exc_info=True)
        return result

    @guarded_transition
    def save_basic_fact(self, user_id, fact, provenance, db):
        """Serialize per user and atomically replace a fixed key before indexing."""
        from sqlalchemy import select
        from src.web_app.models.orm import Memory, User
        from src.web_app.memory.basic_facts import LABELS, content_for
        from src.web_app.services.deletion_guard import check_conversation
        if runtime_policy(db).get("writes_blocked"):
            raise ValueError("memory_writes_disabled")
        if fact.get("key") not in LABELS or not fact.get("value"):
            raise ValueError("invalid_basic_fact")
        check_conversation(db, user_id, provenance.get("conversation_id"), require_exists=bool(provenance.get("conversation_id")))
        provenance = dict(provenance)
        if provenance.get("source_run_id"):
            from src.web_app.models.orm import AgentChatMessage
            source = db.scalar(select(AgentChatMessage).where(AgentChatMessage.user_id == user_id,
                AgentChatMessage.run_id == provenance["source_run_id"], AgentChatMessage.role == "user").order_by(AgentChatMessage.id).limit(1))
            if source:
                provenance.update(source_message_id=source.id, source_quote=source.content, source_verified=True)

        db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one()
        rows = MemoryRepository(db).list_by_user(user_id)
        active = [m for m in rows if m.scope == "user" and (m.metadata_json or {}).get("fact_key") == fact["key"]
                  and (m.metadata_json or {}).get("status", "active") == "active"]
        same = next((m for m in active if m.metadata_json.get("fact_value") == fact["value"]), None)
        if same:
            return self._update_existing(same, same.importance, provenance, db)
        meta = {**provenance, "fact_key": fact["key"], "fact_value": fact["value"],
                "category": fact["key"], "confirmed": True, "confidence": 1.0,
                "status": "active", "visible_in_long_term_memory": True, "stability": "long_term"}
        item = Memory(user_id=user_id, content=content_for(fact), memory_type="semantic",
                      importance=0.95, source_type="confirmed_basic_fact", scope="user", scope_id="", metadata_json={**meta,
                          "evidence": evidence(meta), "evidence_count": len({e["message_id"] for e in evidence(meta)})})
        try:
            db.add(item)
            db.flush()
            item.metadata_json = {**item.metadata_json, "supersedes": [old.id for old in active]}
            for old in active:
                old.metadata_json = {**old.metadata_json, "status": "superseded", "superseded_by": item.id}
            db.commit()
            db.refresh(item)
        except Exception:
            db.rollback()
            raise
        return self._index_saved(item, db)

    @guarded_transition
    def restore_memory(self, user_id, item, db):
        """Restoring a fixed-key fact also replaces its current active value."""
        from sqlalchemy import select
        from src.web_app.models.orm import User
        meta = dict(item.metadata_json or {})
        if item.user_id != user_id:
            raise ValueError("memory_not_owned")
        from src.web_app.services.deletion_guard import check_conversation
        check_conversation(db, user_id, item.scope_id if item.scope == "conversation" else None, require_exists=item.scope == "conversation")
        key = meta.get("fact_key")
        conflict = db.get(type(item), meta["conflicts_with"]) if meta.get("conflicts_with") else None
        if conflict and conflict.user_id == user_id and conflict.scope == item.scope and conflict.scope_id == item.scope_id:
            key = (conflict.metadata_json or {}).get("fact_key", key)
            meta["fact_key"] = key
            meta["supersedes"] = conflict.id
        if key:
            db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one()
            for other in MemoryRepository(db).list_by_user(user_id):
                previous = other.metadata_json or {}
                if other.id != item.id and other.scope == item.scope and other.scope_id == item.scope_id and previous.get("fact_key") == key and previous.get("status", "active") == "active":
                    other.metadata_json = {**previous, "status": "superseded", "superseded_by": item.id}
        meta.pop("superseded_by", None)
        item.metadata_json = {**meta, "status": "active", "confirmed": True}
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        return self._index_saved(item, db)

    @guarded_transition
    def edit_memory(self, user_id, item, payload, db):
        if item.user_id != user_id:
            raise ValueError("memory_not_owned")
        from src.web_app.services.deletion_guard import check_conversation
        check_conversation(db, user_id, item.scope_id if item.scope == "conversation" else None, require_exists=item.scope == "conversation")
        meta = dict(item.metadata_json or {})
        safe = {"category", "sensitive", "visible_in_long_term_memory", "stability"}
        meta.update({k: v for k, v in (payload.get("metadata") or {}).items() if k in safe})
        if "status" in payload:
            if payload["status"] not in {"active", "archived", "pending", "low_confidence", "superseded"}:
                raise ValueError("invalid_memory_status")
            if payload["status"] == "active" and meta.get("status") != "active":
                return self.restore_memory(user_id, item, db)
            meta["status"] = payload["status"]
        values = {"metadata_json": meta}
        if "importance" in payload:
            importance = float(payload["importance"])
            if not 0 <= importance <= 1:
                raise ValueError("invalid_importance")
            values["importance"] = importance
        if "content" in payload:
            content = str(payload["content"]).strip()
            if not content:
                raise ValueError("empty_memory_content")
            if content != item.content:
                # An authenticated edit targets this exact fact. No model comparison
                # or scope reclassification is needed for an explicit correction.
                from sqlalchemy import select
                from src.web_app.models.orm import Memory, User
                db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one()
                reviewed = {**meta, "confirmed": True, "supersedes": item.id,
                    "manual_evidence": {"quote": content, "reviewed_at": datetime.now(UTC).isoformat()}}
                reviewed.pop("superseded_by", None)
                reviewed.pop("is_summary", None)
                reviewed.pop("source_memory_ids", None)
                reviewed.pop("fact_value", None)
                replacement = Memory(user_id=user_id, content=content, scope=item.scope, scope_id=item.scope_id,
                    memory_type=item.memory_type, importance=values.get("importance", item.importance),
                    source_type="manual_edit", metadata_json=reviewed)
                try:
                    db.add(replacement)
                    db.flush()
                    item.metadata_json = {**(item.metadata_json or {}), "status": "superseded", "superseded_by": replacement.id}
                    db.commit()
                    db.refresh(replacement)
                except Exception:
                    db.rollback()
                    raise
                return self._index_saved(replacement, db)
        MemoryRepository(db).update(item, **values)
        return self._index_saved(item, db)

    @guarded_transition
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
        metadata = dict(metadata or {})
        if db and runtime_policy(db).get("writes_blocked"):
            raise ValueError("memory_writes_disabled")
        if not db:
            return self.add_memory(user_id, content, memory_type, importance, source_type, metadata, db)
        from sqlalchemy import select
        from src.web_app.models.orm import Memory, User, AgentRun
        from src.web_app.agent.runtime.chat_control import execution
        from src.web_app.services.deletion_guard import check_conversation
        metadata = validate_source(db, user_id, metadata)
        token = execution.get()
        run_id = metadata.get("run_id") or metadata.get("agent_run_id") or (token.run_id if token else None)
        run = db.get(AgentRun, int(run_id)) if run_id and str(run_id).isdigit() else None
        if run and run.user_id == user_id:
            metadata.update(run_id=run.id, conversation_id=run.conversation_id)
        scope, scope_id = infer_scope(metadata, metadata.get("conversation_id", ""))
        check_conversation(db, user_id, metadata.get("conversation_id"), require_exists=bool(metadata.get("conversation_id")))
        db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one()
        metadata["fact_key"] = fact_key(content, metadata)
        rows = MemoryRepository(db).search_by_type(user_id, memory_type, 0)
        candidates = [m for m in rows if (m.scope, m.scope_id) == (scope, scope_id)
                      and (m.metadata_json or {}).get("status", "active") == "active"
                      and not (m.metadata_json or {}).get("is_summary")]
        plausible = [m for m in candidates if normalized(m.content) == normalized(content)
            or (m.metadata_json or {}).get("fact_key") == metadata["fact_key"]
            or ((m.metadata_json or {}).get("category", "") == metadata.get("category", "")
                and (self._similarity(content, m.content) >= .55 or
                    (bool(metadata.get("category")) and (not metadata.get("entity") or (m.metadata_json or {}).get("entity") == metadata.get("entity")))))]
        plausible.sort(key=lambda m: (normalized(m.content) == normalized(content),
            (m.metadata_json or {}).get("fact_key") == metadata["fact_key"], self._similarity(content, m.content)), reverse=True)
        old, relation, uncertain = None, "unrelated", None
        for candidate in plausible[:4]:
            compared = compare(candidate.content, content,
                same_key=(candidate.metadata_json or {}).get("fact_key") == metadata["fact_key"])
            if compared == "ambiguous" and uncertain is None:
                uncertain = candidate
            if compared in {"duplicate", "supplement", "correction"}:
                old, relation = candidate, compared
                break
        if old is None and uncertain is not None:
            old, relation = uncertain, "ambiguous"
        if old and (metadata.get("status", "active") != "active" or (relation != "duplicate" and
            source_order(old.metadata_json or {}) > source_order(metadata) and source_order(metadata) != (0, 0))):
            relation = "ambiguous"
        if relation == "duplicate":
            return self._update_existing(old, importance, metadata, db)
        if relation == "ambiguous":
            metadata.update(status="pending", conflicts_with=old.id)
            pending = next((m for m in rows if m.content == content and m.scope == scope and m.scope_id == scope_id
                            and (m.metadata_json or {}).get("status") == "pending"), None)
            if pending:
                return {**self._to_dict(pending), "ok": True, "deduped": True, "qdrant_indexed": False}
        item = Memory(user_id=user_id, content=content, memory_type=memory_type, importance=importance,
                      source_type=source_type, scope=scope, scope_id=scope_id,
                      metadata_json={**metadata, "evidence": evidence(metadata), "visible_in_long_term_memory": memory_type in self._QDRANT_MEMORY_TYPES})
        item.metadata_json = {**item.metadata_json, "evidence_count": len({e["message_id"] for e in evidence(metadata)})}
        if relation == "supplement":
            item.content = old.content + "\n" + content
            combined = evidence({"evidence": evidence(old.metadata_json or {}) + evidence(metadata)})
            item.metadata_json = {**item.metadata_json, "fact_key": (old.metadata_json or {}).get("fact_key", metadata["fact_key"]),
                                  "evidence": combined, "evidence_count": len({e["message_id"] for e in combined})}
        if relation in {"correction", "supplement"}:
            item.metadata_json = {**item.metadata_json, "supersedes": old.id}
        try:
            db.add(item)
            db.flush()
            if relation in {"correction", "supplement"}:
                old.metadata_json = {**(old.metadata_json or {}), "status": "superseded", "superseded_by": item.id}
            db.commit()
            db.refresh(item)
        except Exception:
            db.rollback()
            raise
        result = self._index_saved(item, db)
        result["updated_existing"] = relation in {"correction", "supplement"}
        return result

    def _find_similar(self, user_id, content, memory_type, db):
        return next((m for m in MemoryRepository(db).search_by_type(user_id, memory_type, .3)
                     if (m.metadata_json or {}).get("status", "active") == "active"
                     and normalized(m.content) == normalized(content)), None)

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
        metadata = dict(metadata or {})
        merged_evidence = evidence({"evidence": evidence(current_meta) + evidence(metadata)})
        if source_order(metadata) < source_order(current_meta):
            for key in ("run_id", "source_run_id", "source_message_id", "source_quote", "conversation_id", "source_verified"):
                if key in current_meta:
                    metadata[key] = current_meta[key]
        evidence_count = len({e["message_id"] for e in merged_evidence}) or current_meta.get("evidence_count", 1)
        updated_importance = max(existing.importance, importance)
        updated_meta = {
            **current_meta,
            **(metadata or {}),
            "last_seen_at": datetime.now(UTC).isoformat(),
            "evidence_count": evidence_count,
            "evidence": merged_evidence,
            "updated_from": existing.importance,
        }
        # Boost importance slightly with repeated evidence
        if evidence_count >= 3 and evidence_count > current_meta.get("evidence_count", 1):
            updated_importance = min(0.98, updated_importance + 0.05)
        repo.update(
            existing,
            importance=updated_importance,
            metadata_json=updated_meta,
        )
        result = self._index_saved(existing, db)
        result.update({
            "deduped": True,
            "updated_existing": True,
        })
        return result

    def search_memory(self, user_id, query="", memory_type=None, min_importance=0.0, db=None,
                      *, memory_types=None, limit=8, conversation_id=None, use_memory=True, management=False):
        """Hybrid retrieval; PostgreSQL decides visibility, ownership and scope."""
        if not use_memory or (db and not management and runtime_policy(db).get("use_memory") is False):
            return []
        types = memory_types or ([memory_type] if memory_type else ["semantic", "episodic", "working"] if management else ["semantic", "episodic"])
        scores = {}
        candidates = {}
        def accept(d):
            if management:
                return d["memory_type"] in types and d.get("importance", 0) >= min_importance and d.get("metadata", {}).get("status") != "deleting"
            if db and d.get("metadata", {}).get("is_summary"):
                ids = d["metadata"].get("source_memory_ids", [])
                sources = MemoryRepository(db).get_by_ids(user_id, ids)
                if not ids or len(sources) != len(ids) or not all(not (m.metadata_json or {}).get("is_summary") and eligible(self._to_dict(m), conversation_id, allow_legacy=management) for m in sources):
                    return False
            return d["memory_type"] in types and d.get("importance", 0) >= min_importance and eligible(d, conversation_id, allow_legacy=management)
        if db:
            repo = MemoryRepository(db)
            if query:
                store = self._get_qdrant_store()
                if store:
                    try:
                        hits = store.search_memory(user_id=user_id, query=query, memory_types=types,
                                                   limit=max(32, limit * 4), score_threshold=.25)
                        scores = {int(h["memory_id"]): h["score"] for h in hits}
                        for m in repo.get_by_ids(user_id=user_id, ids=list(scores)):
                            d = self._to_dict(m)
                            if accept(d):
                                candidates[m.id] = d
                    except Exception:
                        logger.warning("memory.qdrant_search_failed", exc_info=True)
            rows = repo.search_keywords(user_id, query, types, min_importance, conversation_id=conversation_id, management=management) if query else repo.search(user_id, min_importance=min_importance)
            for m in rows:
                d = self._to_dict(m)
                if accept(d):
                    candidates[m.id] = d
        else:
            for d in self._items:
                if d["user_id"] == user_id and (not query or query.casefold() in d["content"].casefold()) and accept(d):
                    candidates[d["id"]] = d
        self._last_qdrant_hits = len(scores)
        self._last_search_backend = "hybrid" if scores else "postgres_keywords" if candidates else "no_results"
        def recency(d):
            value = (d.get("metadata") or {}).get("last_seen_at") or d.get("updated_at")
            try:
                stamp = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=UTC)
                return 1 / (1 + max(0, (datetime.now(UTC) - stamp).total_seconds() / 86400) / 30)
            except (TypeError, AttributeError, ValueError):
                return 0
        ranked = sorted(candidates.values(), key=lambda d: (.65 * scores.get(d["id"], 0) + .25 * d.get("importance", 0) + .10 * recency(d), d["id"]), reverse=True)
        for d in ranked:
            if d["id"] in scores:
                d["_qdrant_score"] = scores[d["id"]]
        return ranked[:limit]

    def is_recallable(self, item, db, conversation_id=None):
        d = self._to_dict(item)
        if not eligible(d, conversation_id):
            return False
        if d["metadata"].get("is_summary"):
            ids = d["metadata"].get("source_memory_ids", [])
            sources = MemoryRepository(db).get_by_ids(item.user_id, ids)
            return bool(ids) and len(sources) == len(ids) and all(
                not (m.metadata_json or {}).get("is_summary") and eligible(self._to_dict(m), conversation_id) for m in sources)
        return True

    def get_baseline_memories(
        self,
        user_id: int,
        db: Session | None = None,
        *,
        categories: set[str] | list[str] | tuple[str, ...] | None = None,
        min_importance: float = 0.75,
        limit: int = 6,
        use_memory: bool = True,
    ) -> list[dict[str, Any]]:
        """Read stable personal preferences without using the current query.

        Read-only by contract: no writes, no consolidation, no importance updates.
        """
        from src.web_app.memory.basic_facts import LABELS
        if not use_memory or (db and runtime_policy(db).get("use_memory") is False):
            return []
        allowed = set(categories or {
            "name_preference", "language_preference", "tone_preference", "answer_preference", "output_preference", "preference", "negative_preference",
            *LABELS,
        })
        items = ([self._to_dict(m) for m in MemoryRepository(db).search_by_type(user_id, "semantic", min_importance)]
                 if db else [m for m in self._items if m.get("user_id") == user_id and m.get("memory_type") == "semantic"])
        accepted = []
        for item in items:
            meta = item.get("metadata") or {}
            if (not eligible(item) or float(item.get("importance") or 0) < min_importance
                or meta.get("status", "active") != "active"
                or not meta.get("visible_in_long_term_memory", True)
                or float(meta.get("confidence", 1) or 0) < 0.55
                or meta.get("category") not in allowed):
                continue
            if meta.get("category") in LABELS and not meta.get("confirmed"):
                continue
            accepted.append(item)
        return sorted(accepted, key=lambda m: (not bool(m["metadata"].get("confirmed") and m["metadata"].get("fact_key") in LABELS),
                      -float(m["importance"]), -int(m["id"])))[:limit]

    def get_semantic_memories(self, user_id: int, db: Session, min_importance: float = 0.3) -> list[dict[str, Any]]:
        from src.web_app.memory.policy import settings_for
        return self.search_memory(user_id, memory_type="semantic", min_importance=min_importance,
            db=db, limit=1000, use_memory=settings_for(db, user_id)["use_memory"])

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

    @guarded_transition
    def consolidate_memory(self, user_id, db=None, *, conversation_id=None):
        """Build scoped summaries without deleting evidence or promoting stale records."""
        if not db:
            return {"user_id": user_id, "promoted": 0, "mode": "mock"}
        repo = MemoryRepository(db)
        groups = {}
        for m in repo.list_by_user(user_id):
            d = self._to_dict(m)
            if eligible(d, conversation_id) and not d["metadata"].get("is_summary"):
                groups.setdefault((m.scope, m.scope_id), []).append(m)
        summaries = []
        for (scope, scope_id), rows in groups.items():
            content = "\n".join(m.content for m in sorted(rows, key=lambda m: m.id))
            ids = sorted(m.id for m in rows)
            meta = {"category": "scope_summary", "is_summary": True, "source_memory_ids": ids,
                    "status": "active", "visible_in_long_term_memory": True, "confidence": 1.0}
            old = next((m for m in repo.list_by_user(user_id) if m.scope == scope and m.scope_id == scope_id and (m.metadata_json or {}).get("is_summary")), None)
            if old:
                repo.update(old, content=content, metadata_json=meta)
                result = self._index_saved(old, db)
            else:
                from src.web_app.models.orm import Memory
                item = repo.create(user_id=user_id, content=content, memory_type="semantic", importance=.75,
                                   scope=scope, scope_id=scope_id, metadata_json=meta, source_type="memory_summary")
                result = self._index_saved(item, db)
            summaries.append(result)
        return {"user_id": user_id, "summaries": summaries, "summary_count": len(summaries), "archived": 0, "total_promoted": 0,
                "promoted_working_to_episodic": 0, "promoted_episodic_to_semantic": 0}

    def forget_memory(self, user_id: int, memory_id: int | None = None, db: Session | None = None) -> dict[str, Any]:
        if db:
            repo = MemoryRepository(db)
            if memory_id:
                item = repo.get_by_id(memory_id)
                if item and item.user_id == user_id:
                    graph_warning = self._mark_memory_graph(user_id, item.id, "forgotten", "forget_memory")
                    warning = self._delete_memory_vector(item.id)
                    db.delete(item)
                    db.commit()
                    result = {"deleted": 1}
                    if warning:
                        result["vector_cleanup_warning"] = warning
                    if graph_warning:
                        result["graph_warning"] = graph_warning
                    return result
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
        self._mark_memory_graph(memory.user_id, memory.id, "archived", reason)
        self._delete_memory_vector(memory.id)
        return self._to_dict(memory)

    def _sync_memory_graph(self, user_id: int, memory: Any) -> dict[str, Any]:
        try:
            from src.web_app.graph.memory_projector import memory_graph_projector
            return memory_graph_projector.sync_memory(user_id=user_id, memory=memory)
        except Exception as exc:
            logger.warning("memory.graph_sync_failed user_id=%s error=%s", user_id, exc, exc_info=True)
            return {"synced": False, "warning": str(exc)[:200]}

    def _mark_memory_graph(self, user_id: int, memory_id: int | str, status: str, reason: str) -> str | None:
        try:
            from src.web_app.graph.memory_projector import memory_graph_projector
            result = memory_graph_projector.mark_memory_status(
                user_id=user_id,
                memory_id=memory_id,
                status=status,
                reason=reason,
            )
            return result.get("warning")
        except Exception as exc:
            logger.warning(
                "memory.graph_status_failed user_id=%s memory_id=%s error=%s",
                user_id,
                memory_id,
                exc,
                exc_info=True,
            )
            return str(exc)[:200]

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
        save_results: list[dict[str, Any]] = []
        filtered_out = {"episodic": 0, "semantic": 0}
        now_ts = datetime.now(UTC).isoformat()
        provenance = {}
        source_text = ""
        if db and run_id and str(run_id).isdigit():
            from sqlalchemy import select
            from src.web_app.models.orm import AgentRun, AgentChatMessage
            run = db.get(AgentRun, int(run_id))
            if not run or run.user_id != user_id:
                raise ValueError("invalid_memory_source")
            source = db.scalar(select(AgentChatMessage).where(AgentChatMessage.run_id == run.id,
                AgentChatMessage.user_id == user_id, AgentChatMessage.role == "user").order_by(AgentChatMessage.id).limit(1))
            source_text = source.content if source else run.user_input
            provenance = {"run_id": run.id, "conversation_id": run.conversation_id,
                          "source_message_id": source.id if source else None}
        def source_metadata(mem):
            quote = mem.get("source_quote", "")
            verified = bool(quote and provenance.get("source_message_id") and quote in source_text)
            return {**provenance, "source_quote": quote, "source_verified": verified,
                    "entity": mem.get("entity", ""), "personal_long_term": bool(mem.get("personal_long_term")),
                    **({"confirmed": True, "confirmation": "verified_source"} if verified and mem.get("category") in {"preferred_name", "response_language", "script_preference"} else {}),
                    **({"status": "pending"} if not verified else {})}

        # working: low-barrier, visible_in_long_term_memory=False
        for mem in extraction.get("working_memories", []):
            result = self.add_memory(user_id, mem["content"], memory_type="working",
                importance=mem.get("importance", 0.3),
                metadata={"run_id": run_id, "category": mem.get("category", ""), "source": mem.get("source", ""),
                          "visible_in_long_term_memory": False, "stability": "temporary",
                          "status": "active", "confidence": mem.get("confidence", 0.95)}, db=db)
            saved["working"].append(result)
            save_results.append(result)

        # episodic implicit auto-write: high confidence only
        for mem in extraction.get("episodic_memories", []):
            importance = mem.get("importance", 0.5)
            confidence = mem.get("confidence", 0.80)
            if importance < 0.85 or confidence < 0.80:
                filtered_out["episodic"] += 1; continue
            result = self.add_with_dedup(user_id, mem["content"], memory_type="episodic",
                importance=importance,
                metadata={"category": mem.get("category", ""), "source": mem.get("source", ""),
                          "visible_in_long_term_memory": True, "stability": mem.get("stability", "medium_term"),
                          "status": mem.get("status", "active"), "evidence_count": 1, "last_seen_at": now_ts, "confidence": confidence, **source_metadata(mem)}, db=db)
            if result:
                saved["episodic"].append(result)
                save_results.append(result)

        # semantic implicit auto-write: high confidence only
        for mem in extraction.get("semantic_memories", []):
            importance = mem.get("importance", 0.8)
            confidence = mem.get("confidence", 0.80)
            if importance < 0.88 or confidence < 0.85:
                filtered_out["semantic"] += 1; continue
            result = self.add_with_dedup(user_id, mem["content"], memory_type="semantic",
                importance=importance,
                metadata={"category": mem.get("category", ""), "source": mem.get("source", "home_chat"),
                          "visible_in_long_term_memory": True, "stability": mem.get("stability", "long_term"),
                          "status": mem.get("status", "active"), "evidence_count": 1, "last_seen_at": now_ts, "confidence": confidence, **source_metadata(mem)}, db=db)
            if result:
                saved["semantic"].append(result)
                save_results.append(result)

        total_saved = len(saved["working"]) + len(saved["episodic"]) + len(saved["semantic"])
        qdrant_indexed_count = sum(1 for r in save_results if r.get("qdrant_indexed"))
        logger.info("memory.extract_and_save: run_id=%s working=%d episodic=%d semantic=%d filtered_episodic=%d filtered_semantic=%d qdrant_indexed=%d",
                    run_id, len(saved["working"]), len(saved["episodic"]), len(saved["semantic"]),
                    filtered_out["episodic"], filtered_out["semantic"], qdrant_indexed_count)
        return {"extraction": extraction, "saved": saved, "save_results": save_results, "filtered_out": filtered_out, "total_saved": total_saved, "qdrant_indexed_count": qdrant_indexed_count}

    def _to_dict(self, item) -> dict[str, Any]:
        return {
            "id": item.id,
            "user_id": item.user_id,
            "content": item.content,
            "memory_type": item.memory_type,
            "importance": item.importance,
            "metadata": item.metadata_json or {},
            "scope": item.scope, "scope_id": item.scope_id,
            "expires_at": item.expires_at.isoformat() if item.expires_at else None,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        }

    def _to_summary(self, item) -> dict[str, Any]:
        content = "[masked sensitive memory]" if (item.metadata_json or {}).get("sensitive") else item.content
        return {"id": item.id, "memory_type": item.memory_type, "content": content, "importance": item.importance}


memory_service = MemoryService()
