from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import Any

from src.web_app.core.config import settings
from src.web_app.graph.neo4j_client import Neo4jUnavailable
from src.web_app.graph.repositories import GraphRepository

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-./#]{1,40}|[\u4e00-\u9fff]{2,12}")
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "you", "your", "用户", "当前", "这个",
    "需要", "希望", "进行", "不要", "可以", "项目", "文档", "记忆",
}
_KNOWN_TECH_TERMS = {
    "agent", "rag", "qdrant", "postgresql", "postgres", "redis", "neo4j", "fastapi",
    "langgraph", "langchain", "python", "dashscope", "bm25", "hybrid", "qdrant/bm25",
}
_GOAL_CATEGORIES = {"project_goal", "workflow_pattern", "research_preference"}
_PREFERENCE_CATEGORIES = {
    "preference", "negative_preference", "answer_preference", "name_preference",
    "language_preference", "tone_preference", "tool_preference", "document_preference",
}
_BOUNDARY_CATEGORIES = {"boundary"}


class MemoryGraphProjector:
    def __init__(self, repository: GraphRepository | None = None):
        self.repository = repository or GraphRepository()

    def sync_memory(self, *, user_id: int, memory: Any) -> dict[str, Any]:
        if not self._enabled():
            return {"enabled": False, "synced": False, "reason": "disabled"}
        try:
            payload = self._memory_payload(user_id, memory)
            meta = payload["metadata"]
            category = str(meta.get("category", "") or "")
            content = str(payload["content"] or "")
            terms = _extract_terms(content, meta)
            topics = terms[:8]
            goals = terms[:4] if category in _GOAL_CATEGORIES else []
            preferences = terms[:4] if category in _PREFERENCE_CATEGORIES else []
            boundaries = terms[:4] if category in _BOUNDARY_CATEGORIES else []
            self.repository.upsert_memory_projection(
                user_id=user_id,
                memory=payload["node"],
                topics=topics,
                goals=goals,
                preferences=preferences,
                boundaries=boundaries,
                project_key=settings.neo4j_project_key,
            )
            return {"enabled": True, "synced": True, "topics": topics}
        except Neo4jUnavailable as exc:
            logger.warning("graph.memory_sync_unavailable user_id=%s reason=%s", user_id, exc)
            return {"enabled": True, "synced": False, "warning": str(exc)}
        except Exception as exc:
            logger.warning("graph.memory_sync_failed user_id=%s error=%s", user_id, exc, exc_info=True)
            return {"enabled": True, "synced": False, "warning": str(exc)[:200]}

    def mark_memory_status(
        self,
        *,
        user_id: int,
        memory_id: int | str,
        status: str,
        reason: str = "",
    ) -> dict[str, Any]:
        if not self._enabled():
            return {"enabled": False, "synced": False, "reason": "disabled"}
        try:
            self.repository.mark_memory_status(user_id=user_id, memory_id=memory_id, status=status, reason=reason)
            return {"enabled": True, "synced": True}
        except Neo4jUnavailable as exc:
            logger.warning("graph.memory_status_unavailable user_id=%s memory_id=%s reason=%s", user_id, memory_id, exc)
            return {"enabled": True, "synced": False, "warning": str(exc)}
        except Exception as exc:
            logger.warning("graph.memory_status_failed user_id=%s memory_id=%s error=%s", user_id, memory_id, exc, exc_info=True)
            return {"enabled": True, "synced": False, "warning": str(exc)[:200]}

    def _enabled(self) -> bool:
        return bool(settings.enable_neo4j and settings.neo4j_memory_graph_enabled)

    def _memory_payload(self, user_id: int, memory: Any) -> dict[str, Any]:
        if isinstance(memory, dict):
            memory_id = memory.get("id") or memory.get("memory_id")
            content = memory.get("content", "")
            memory_type = memory.get("memory_type", "working")
            importance = memory.get("importance", 0.0)
            meta = memory.get("metadata") or memory.get("metadata_json") or {}
            created_at = memory.get("created_at")
            updated_at = memory.get("updated_at")
        else:
            memory_id = getattr(memory, "id")
            content = getattr(memory, "content", "")
            memory_type = getattr(memory, "memory_type", "working")
            importance = getattr(memory, "importance", 0.0)
            meta = getattr(memory, "metadata_json", {}) or {}
            created_at = getattr(memory, "created_at", None)
            updated_at = getattr(memory, "updated_at", None)
        now = datetime.now(UTC).isoformat()
        preview = _preview(content, 180)
        node = {
            "memory_id": str(memory_id),
            "user_id": int(user_id),
            "memory_type": str(memory_type or "working"),
            "category": str(meta.get("category", "") or ""),
            "importance": float(importance or 0.0),
            "confidence": float(meta.get("confidence", 0.95) or 0.95),
            "status": str(meta.get("status", "active") or "active"),
            "preview": preview,
            "content_hash": hashlib.sha256(str(content or "").encode("utf-8")).hexdigest(),
            "source": "memory_projector",
            "scope": "user",
            "project_key": settings.neo4j_project_key,
            "created_at": _iso(created_at) or now,
            "updated_at": _iso(updated_at) or now,
        }
        return {"node": node, "metadata": meta, "content": content}


def _extract_terms(content: str, metadata: dict[str, Any]) -> list[str]:
    raw_terms: list[str] = []
    for key in ("topic", "topics", "tags", "technology", "technologies"):
        value = metadata.get(key)
        if isinstance(value, str):
            raw_terms.extend(re.split(r"[,，;；\s]+", value))
        elif isinstance(value, list):
            raw_terms.extend(str(item) for item in value)
    lowered = content.lower()
    for term in _KNOWN_TECH_TERMS:
        if term in lowered:
            raw_terms.append(term)
    raw_terms.extend(_TOKEN_RE.findall(content))
    seen: set[str] = set()
    terms: list[str] = []
    for term in raw_terms:
        key = _normalize_key(term)
        if not key or key in _STOPWORDS or key in seen:
            continue
        seen.add(key)
        terms.append(key)
    return terms


def _normalize_key(value: str) -> str:
    key = str(value or "").strip().lower()
    key = re.sub(r"\s+", "_", key)
    key = re.sub(r"[^a-z0-9_\-./#\u4e00-\u9fff]+", "", key)
    return key[:80]


def _preview(content: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(content or "")).strip()
    return text[:limit]


def _iso(value: Any) -> str:
    if not value:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


memory_graph_projector = MemoryGraphProjector()

