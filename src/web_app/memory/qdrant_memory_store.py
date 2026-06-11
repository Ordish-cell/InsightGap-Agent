"""Qdrant vector store for long-term memory (semantic retrieval layer).

PostgreSQL `memories` is the authoritative source. This store holds embeddings
for semantic search.  On write failures the PG record is NOT rolled back.
On search failures callers fall back to PG ILIKE / recent-important.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.web_app.core.config import settings
from src.web_app.rag.embeddings import embed_text, get_embedding_dimension

# ── Cap memory content before embedding ────────────────────────────────
MAX_MEMORY_EMBED_CHARS = 4000


class QdrantMemoryStore:
    """Thin wrapper over a Qdrant collection for memory embeddings."""

    def __init__(self, client: QdrantClient | None = None):
        if client is not None:
            self._client = client
        else:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=settings.qdrant_timeout,
            )
        self.collection = settings.memory_qdrant_collection
        self._collection_ensured = False

    # ── helpers ────────────────────────────────────────────────────────

    def _vector_size(self) -> int:
        return get_embedding_dimension()

    def _embed_content(self, content: str) -> list[float]:
        text = content.strip()[:MAX_MEMORY_EMBED_CHARS]
        return embed_text(text)

    @staticmethod
    def generate_point_id() -> str:
        """Qdrant Cloud requires UUID or unsigned-integer point IDs."""
        return str(uuid4())

    def _build_payload(
        self,
        memory_id: int | str,
        user_id: int | str,
        content: str,
        memory_type: str,
        importance: float,
        source_type: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "memory_id": str(memory_id),
            "user_id": str(user_id),
            "memory_type": memory_type,
            "content_preview": content.strip()[:500],
            "importance": importance,
            "source_type": source_type or "",
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }

    # ── public API ─────────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        if self._collection_ensured:
            return
        names = [item.name for item in self._client.get_collections().collections]
        if self.collection not in names:
            dist = Distance.COSINE if settings.qdrant_distance.lower() == "cosine" else Distance.COSINE
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self._vector_size(), distance=dist),
            )
        # Ensure payload indexes for filtering (required by Qdrant Cloud)
        for field in ("user_id", "memory_id", "memory_type"):
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass  # index may already exist
        self._collection_ensured = True

    def upsert_memory(
        self,
        memory_id: int | str,
        user_id: int | str,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.0,
        source_type: str = "",
        metadata: dict[str, Any] | None = None,
        point_id: str | None = None,
    ) -> str:
        """Upsert a memory into Qdrant. Returns the Qdrant point ID (UUID)."""
        self.ensure_collection()
        vector = self._embed_content(content)
        payload = self._build_payload(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            source_type=source_type,
            metadata=metadata,
        )
        pid = point_id or self.generate_point_id()
        point = PointStruct(id=pid, vector=vector, payload=payload)
        self._client.upsert(collection_name=self.collection, points=[point])
        return pid

    def search_memory(
        self,
        user_id: int | str,
        query: str,
        memory_types: list[str] | None = None,
        limit: int = 8,
        score_threshold: float = 0.25,
    ) -> list[dict[str, Any]]:
        self.ensure_collection()
        query_vector = self._embed_content(query)
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))]
        if memory_types:
            from qdrant_client.models import MatchAny
            conditions.append(
                FieldCondition(key="memory_type", match=MatchAny(any=memory_types))
            )

        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=Filter(must=conditions),
                with_payload=True,
            )
            hits = response.points
        else:
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=Filter(must=conditions),
                with_payload=True,
            )

        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "memory_id": int(payload.get("memory_id", 0)),
                "score": float(hit.score),
                "payload": {
                    k: v for k, v in payload.items()
                    if k != "metadata"  # keep payload light
                },
            })
        return results

    def delete_by_memory_id(self, memory_id: int | str) -> None:
        """Delete points by memory_id payload filter."""
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="memory_id", match=MatchValue(value=str(memory_id))),
                ]
            ),
        )

    # ── Backfill / consistency helpers ───────────────────────────────

    def memory_exists(self, memory_id: int | str) -> bool:
        """Check whether a point with the given memory_id payload exists."""
        self.ensure_collection()
        conditions = [
            FieldCondition(key="memory_id", match=MatchValue(value=str(memory_id))),
        ]
        if hasattr(self._client, "query_points"):
            resp = self._client.query_points(
                collection_name=self.collection,
                query=[0.0] * self._vector_size(),  # dummy vector
                limit=1,
                query_filter=Filter(must=conditions),
            )
            return len(resp.points) > 0
        else:
            hits = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=conditions),
                limit=1,
            )
            return len(hits[0]) > 0

    def count_indexed_memories(
        self,
        user_id: int | str | None = None,
        memory_types: list[str] | None = None,
    ) -> int:
        """Count indexed memories, optionally filtered by user_id / memory_type."""
        self.ensure_collection()
        # If filters are present, fall back to scroll-based counting since
        # Qdrant Cloud's count endpoint has inconsistent behaviour.
        if user_id is not None or memory_types:
            return len(self.list_indexed_memory_ids(
                user_id=user_id, memory_types=memory_types))
        try:
            info = self._client.get_collection(self.collection)
            return getattr(info, "points_count", 0) or 0
        except Exception:
            return 0

    def list_indexed_memory_ids(
        self,
        user_id: int | str | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10000,
    ) -> set[str]:
        """Return the set of memory_id strings indexed in Qdrant.

        Used for cross-checking PostgreSQL ↔ Qdrant consistency.
        """
        self.ensure_collection()
        conditions: list[FieldCondition] = []
        if user_id is not None:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=str(user_id))))
        if memory_types:
            from qdrant_client.models import MatchAny
            conditions.append(FieldCondition(key="memory_type", match=MatchAny(any=memory_types)))
        ids: set[str] = set()
        offset: str | int | None = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=conditions) if conditions else None,
                limit=min(limit, 1000),
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                mid = payload.get("memory_id", "")
                if mid:
                    ids.add(str(mid))
            if next_offset is None or len(ids) >= limit:
                break
            offset = next_offset
        return ids

