from datetime import UTC, datetime
import hashlib
import inspect
import logging
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.web_app.core.config import settings
from src.web_app.rag.sparse_encoder import build_sparse_document_input, build_sparse_query_input, is_sparse_input_empty

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self):
        if not settings.qdrant_url:
            raise RuntimeError("Qdrant is not configured")
        client_kwargs: dict[str, Any] = {"url": settings.qdrant_url, "api_key": settings.qdrant_api_key or None, "timeout": settings.qdrant_timeout}
        if _qdrant_client_supports_cloud_inference() and settings.qdrant_cloud_inference:
            client_kwargs["cloud_inference"] = True
        self.client = QdrantClient(**client_kwargs)
        self.hybrid_enabled = settings.rag_hybrid_backend == "qdrant_hybrid"
        self.collection = settings.qdrant_hybrid_collection if self.hybrid_enabled else settings.qdrant_collection
        self.dense_vector_name = settings.qdrant_dense_vector_name or "dense"
        self.sparse_vector_name = settings.qdrant_sparse_vector_name or "bm25"
        self.sparse_encoder = (settings.qdrant_sparse_encoder or "hashing_sparse").lower()
        self.sparse_model = settings.qdrant_sparse_model or "Qdrant/bm25"

    def ensure_collection(self) -> None:
        names = [item.name for item in self.client.get_collections().collections]
        if self.collection not in names:
            distance = Distance.COSINE if settings.qdrant_distance.lower() == "cosine" else Distance.COSINE
            if self.hybrid_enabled:
                self.client.create_collection(
                    self.collection,
                    vectors_config={
                        self.dense_vector_name: VectorParams(size=settings.qdrant_vector_size, distance=distance),
                    },
                    sparse_vectors_config={
                        self.sparse_vector_name: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier=Modifier.IDF,
                        ),
                    },
                )
            else:
                self.client.create_collection(self.collection, vectors_config=VectorParams(size=settings.qdrant_vector_size, distance=distance))
        elif self.hybrid_enabled and not self.collection_supports_hybrid():
            raise RuntimeError(f"Qdrant collection {self.collection} does not support dense+sparse hybrid vectors")
        self.ensure_payload_indexes()

    def capability_status(self) -> dict[str, Any]:
        required_models = {
            "Document": Document,
            "SparseVectorParams": SparseVectorParams,
            "Prefetch": Prefetch,
            "FusionQuery": FusionQuery,
            "Fusion": Fusion,
        }
        missing = [name for name, value in required_models.items() if value is None]
        if not _qdrant_client_supports_cloud_inference():
            missing.append("cloud_inference_client_param")
        if self.sparse_encoder == "qdrant_cloud_bm25" and not settings.qdrant_cloud_inference:
            missing.append("cloud_inference_disabled")
        if self.sparse_encoder == "qdrant_cloud_bm25":
            try:
                probe = build_sparse_query_input("capability check")
                if not isinstance(probe, Document) or probe.model != self.sparse_model:
                    missing.append("sparse_model_unavailable")
            except Exception:
                missing.append("sparse_model_unavailable")
        server_version = ""
        try:
            if hasattr(self.client, "get_version"):
                server_version = str(self.client.get_version())
        except Exception:
            server_version = ""
        collection_ok = False
        collection_exists = False
        try:
            names = [item.name for item in self.client.get_collections().collections]
            collection_exists = self.collection in names
            collection_ok = (not collection_exists) or self.collection_supports_hybrid()
        except Exception as exc:
            return {"supported": False, "missing": ["collection_check"], "error": str(exc), "server_version": server_version, "collection": self.collection}
        supported = not missing and hasattr(self.client, "query_points") and collection_ok
        return {
            "supported": supported,
            "missing": missing + ([] if hasattr(self.client, "query_points") else ["query_points"]) + ([] if collection_ok else ["dense_sparse_collection_schema"]),
            "server_version": server_version,
            "collection": self.collection,
            "collection_exists": collection_exists,
            "dense_vector_name": self.dense_vector_name,
            "sparse_vector_name": self.sparse_vector_name,
            "sparse_encoder": self.sparse_encoder,
            "sparse_model": self.sparse_model,
            "cloud_inference": bool(settings.qdrant_cloud_inference),
        }

    def collection_supports_hybrid(self) -> bool:
        try:
            info = self.client.get_collection(self.collection)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None)
            sparse_vectors = getattr(params, "sparse_vectors", None)
            dense_ok = isinstance(vectors, dict) and self.dense_vector_name in vectors
            sparse_ok = isinstance(sparse_vectors, dict) and self.sparse_vector_name in sparse_vectors
            return dense_ok and sparse_ok
        except Exception as exc:
            logger.warning("qdrant.hybrid_schema_check_failed collection=%s error=%s", self.collection, exc)
            return False

    def ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes for fields used in filtering.

        Idempotent — ignores errors from indexes that already exist.
        Called after ensure_collection so both new and existing collections
        get the indexes.
        """
        import logging
        _logger = logging.getLogger(__name__)
        for field in ("user_id", "document_id", "file_type", "created_at"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                err_msg = str(exc).lower()
                if "already exists" in err_msg or "already" in err_msg:
                    continue
                _logger.warning("Failed to create Qdrant payload index for %s: %s", field, exc)

    def upsert_chunks(self, user_id: int, document_id: int, chunks: list[dict[str, Any]], vectors: list[list[float]], document: Any) -> list[str]:
        self.ensure_collection()
        point_ids: list[str] = []
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = str(uuid4())
            point_ids.append(point_id)
            content = chunk["content"]
            metadata = dict(chunk.get("metadata", {}))
            chunk_id = str(metadata.get("chunk_id") or point_id)
            content_hash = metadata.get("content_hash") or hashlib.sha256(content.encode("utf-8")).hexdigest()
            payload = {
                "user_id": str(user_id),
                "document_id": str(document_id),
                "chunk_id": chunk_id,
                "qdrant_point_id": point_id,
                "chunk_index": chunk["chunk_index"],
                "content": content,
                "content_preview": content[:500],
                "source_title": document.filename,
                "source_url": None,
                "filename": document.filename,
                "file_type": document.file_type,
                "mime_type": document.file_type,
                "heading_path": chunk.get("heading_path", []),
                "token_count": chunk.get("token_count", 0),
                "content_hash": content_hash,
                "chunk_role": metadata.get("chunk_role", "child"),
                "chunk_type": metadata.get("chunk_type", "text"),
                "parent_id": metadata.get("parent_id"),
                "page_number": metadata.get("page_number"),
                "sheet_name": metadata.get("sheet_name"),
                "header": metadata.get("header", []),
                "row_start": metadata.get("row_start"),
                "row_end": metadata.get("row_end"),
                "created_at": datetime.now(UTC).isoformat(),
                "metadata": metadata,
            }
            point_vector: Any = vector
            if self.hybrid_enabled:
                sparse = build_sparse_document_input(content)
                point_vector = {self.dense_vector_name: vector}
                if not is_sparse_input_empty(sparse):
                    point_vector[self.sparse_vector_name] = sparse
            points.append(PointStruct(id=point_id, vector=point_vector, payload=payload))
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
                using=self.dense_vector_name if self.hybrid_enabled else None,
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
                    "qdrant_point_id": payload.get("qdrant_point_id") or str(hit.id),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": float(hit.score),
                    "content_preview": payload.get("content_preview", ""),
                    "content": payload.get("content", ""),
                    "source_title": payload.get("source_title", ""),
                    "source_url": payload.get("source_url"),
                    "filename": payload.get("filename", ""),
                    "file_type": payload.get("file_type") or payload.get("mime_type", ""),
                    "token_count": payload.get("token_count", 0),
                    "content_hash": payload.get("content_hash", ""),
                    "chunk_role": payload.get("chunk_role", "child"),
                    "chunk_type": payload.get("chunk_type", ""),
                    "parent_id": payload.get("parent_id"),
                    "page_number": payload.get("page_number"),
                    "sheet_name": payload.get("sheet_name"),
                    "metadata": payload.get("metadata", {}),
                    "heading_path": payload.get("heading_path", []),
                }
            )
        return results

    def search_hybrid(self, user_id: int, query_vector: list[float], query_text: str, top_k: int = 5, min_score: float = 0.2, document_ids: list[int] | None = None) -> list[dict[str, Any]]:
        self.ensure_collection()
        status = self.capability_status()
        if not status.get("supported"):
            raise RuntimeError(f"Qdrant hybrid is not supported: {status}")
        sparse = build_sparse_query_input(query_text)
        if is_sparse_input_empty(sparse):
            raise RuntimeError("Sparse query vector is empty")
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))]
        if document_ids:
            conditions.append(FieldCondition(key="document_id", match=MatchAny(any=[str(item) for item in document_ids])))
        query_filter = Filter(must=conditions)
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=query_vector, using=self.dense_vector_name, filter=query_filter, limit=max(top_k * 3, top_k), score_threshold=min_score),
                Prefetch(query=sparse, using=self.sparse_vector_name, filter=query_filter, limit=max(top_k * 3, top_k)),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        results = self._hits_to_results(response.points)
        for item in results:
            item["retrieval_source"] = "qdrant_hybrid"
            item["vector_score"] = 0.0
            item["bm25_score"] = 0.0
            item["sparse_score"] = 0.0
            item["final_score"] = item.get("score", 0.0)
        return results

    def _hits_to_results(self, hits: list[Any]) -> list[dict[str, Any]]:
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "document_id": payload.get("document_id"),
                    "chunk_id": payload.get("chunk_id") or str(hit.id),
                    "qdrant_point_id": payload.get("qdrant_point_id") or str(hit.id),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": float(hit.score),
                    "content_preview": payload.get("content_preview", ""),
                    "content": payload.get("content", ""),
                    "source_title": payload.get("source_title", ""),
                    "source_url": payload.get("source_url"),
                    "filename": payload.get("filename", ""),
                    "file_type": payload.get("file_type") or payload.get("mime_type", ""),
                    "token_count": payload.get("token_count", 0),
                    "content_hash": payload.get("content_hash", ""),
                    "chunk_role": payload.get("chunk_role", "child"),
                    "chunk_type": payload.get("chunk_type", ""),
                    "parent_id": payload.get("parent_id"),
                    "page_number": payload.get("page_number"),
                    "sheet_name": payload.get("sheet_name"),
                    "metadata": payload.get("metadata", {}),
                    "heading_path": payload.get("heading_path", []),
                }
            )
        return results

    def delete_document(self, user_id: int, document_id: int) -> None:
        names = [item.name for item in self.client.get_collections().collections]
        if self.collection not in names:
            logger.info("qdrant.delete_document_skipped_missing_collection collection=%s user_id=%s document_id=%s", self.collection, user_id, document_id)
            return
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


def _qdrant_client_supports_cloud_inference() -> bool:
    try:
        parameters = inspect.signature(QdrantClient).parameters
        return "cloud_inference" in parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    except Exception:
        return False
