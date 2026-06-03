from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams

from src.web_app.core.config import settings


class QdrantVectorStore:
    def __init__(self):
        if not settings.qdrant_url:
            raise RuntimeError("Qdrant is not configured")
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=settings.qdrant_timeout)
        self.collection = settings.qdrant_collection

    def ensure_collection(self) -> None:
        names = [item.name for item in self.client.get_collections().collections]
        if self.collection in names:
            return
        distance = Distance.COSINE if settings.qdrant_distance.lower() == "cosine" else Distance.COSINE
        self.client.create_collection(self.collection, vectors_config=VectorParams(size=settings.qdrant_vector_size, distance=distance))

    def upsert_chunks(self, user_id: int, document_id: int, chunks: list[dict[str, Any]], vectors: list[list[float]], document: Any) -> list[str]:
        self.ensure_collection()
        point_ids: list[str] = []
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = str(uuid4())
            point_ids.append(point_id)
            payload = {
                "user_id": str(user_id),
                "document_id": str(document_id),
                "chunk_id": "",
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "content_preview": chunk["content"][:500],
                "source_title": document.filename,
                "source_url": None,
                "filename": document.filename,
                "mime_type": document.file_type,
                "heading_path": chunk.get("heading_path", []),
                "created_at": datetime.now(UTC).isoformat(),
                "metadata": chunk.get("metadata", {}),
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return point_ids

    def search(self, user_id: int, query_vector: list[float], top_k: int = 5, min_score: float = 0.2, document_ids: list[int] | None = None) -> list[dict[str, Any]]:
        self.ensure_collection()
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))]
        if document_ids:
            conditions.append(FieldCondition(key="document_id", match=MatchAny(any=[str(item) for item in document_ids])))
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                score_threshold=min_score,
                query_filter=Filter(must=conditions),
                with_payload=True,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=min_score,
                query_filter=Filter(must=conditions),
                with_payload=True,
            )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "document_id": payload.get("document_id"),
                    "chunk_id": payload.get("chunk_id") or str(hit.id),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": float(hit.score),
                    "content_preview": payload.get("content_preview", ""),
                    "content": payload.get("content", ""),
                    "source_title": payload.get("source_title", ""),
                    "source_url": payload.get("source_url"),
                    "metadata": payload.get("metadata", {}),
                    "heading_path": payload.get("heading_path", []),
                }
            )
        return results

    def delete_document(self, user_id: int, document_id: int) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=str(user_id))),
                    FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                ]
            ),
        )

    def get_collection_stats(self) -> dict[str, Any]:
        self.ensure_collection()
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points_count": getattr(info, "points_count", 0) or 0,
            "vectors_count": getattr(info, "vectors_count", None) or getattr(info, "indexed_vectors_count", 0) or 0,
        }
