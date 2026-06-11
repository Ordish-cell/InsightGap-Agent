from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.web_app.models.orm import Document
from src.web_app.db.repositories.document_repository import DocumentChunkRepository
from src.web_app.rag.bm25 import BM25Document, bm25_search
from src.web_app.rag.query_analyzer import QueryAnalysis, analyze_query
from src.web_app.rag.reranker import rerank_results
from src.web_app.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class ParentChildRetriever:
    """Vector retrieval over child chunks with PostgreSQL parent context lookup."""

    def __init__(self, db: Session, vector_store: QdrantVectorStore | None = None):
        self.db = db
        self.vector_store = vector_store

    def search(
        self,
        *,
        user_id: int,
        query: str = "",
        query_vector: list[float] | None = None,
        top_k: int = 5,
        min_score: float = 0.2,
        document_ids: list[int] | None = None,
        bm25_candidate_limit: int = 1000,
        backend: str = "python_bm25",
        allow_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        analysis = analyze_query(query)
        if backend == "qdrant_hybrid":
            try:
                return self._qdrant_hybrid_search(user_id, query, query_vector, top_k, min_score, document_ids, analysis)
            except Exception as exc:
                logger.warning("qdrant_hybrid_failed fallback_to_python_bm25 user_id=%s error=%s", user_id, exc, exc_info=True)
                if not allow_fallback:
                    raise
                fallback_results = self._python_bm25_hybrid_search(user_id, query, query_vector, top_k, min_score, document_ids, bm25_candidate_limit, analysis)
                for item in fallback_results:
                    item["retrieval_warning"] = f"qdrant_hybrid_failed: {exc}"
                return fallback_results
        return self._python_bm25_hybrid_search(user_id, query, query_vector, top_k, min_score, document_ids, bm25_candidate_limit, analysis)

    def _qdrant_hybrid_search(
        self,
        user_id: int,
        query: str,
        query_vector: list[float] | None,
        top_k: int,
        min_score: float,
        document_ids: list[int] | None,
        analysis: QueryAnalysis,
    ) -> list[dict[str, Any]]:
        if not self.vector_store or not query_vector:
            raise RuntimeError("Qdrant hybrid requires vector store and dense query vector")
        hits = self.vector_store.search_hybrid(
            user_id=user_id,
            query_vector=query_vector,
            query_text=query,
            top_k=max(top_k * 2, top_k),
            min_score=min_score,
            document_ids=document_ids,
        )
        if not hits:
            return []
        for hit in hits:
            hit.setdefault("retrieval_source", "qdrant_hybrid")
            hit.setdefault("query_type", analysis.query_type)
            hit.setdefault("final_score", hit.get("score", 0.0))
            hit.setdefault("matched_terms", [])
        ranked = rerank_results(hits, analysis, document_ids=document_ids, top_k=top_k)
        for item in ranked:
            item["retrieval_source"] = "qdrant_hybrid"
        return self._enrich_parent_context(user_id, ranked, analysis)

    def _python_bm25_hybrid_search(
        self,
        user_id: int,
        query: str,
        query_vector: list[float] | None,
        top_k: int,
        min_score: float,
        document_ids: list[int] | None,
        bm25_candidate_limit: int,
        analysis: QueryAnalysis,
    ) -> list[dict[str, Any]]:
        vector_hits = self._vector_search(user_id, query_vector, top_k, min_score, document_ids)
        bm25_hits = self._bm25_search(user_id, query, max(top_k * 4, top_k), document_ids, bm25_candidate_limit)
        merged = self._merge_hits(vector_hits, bm25_hits, analysis)
        if not merged:
            return []
        ranked = rerank_results(merged, analysis, document_ids=document_ids, top_k=top_k)
        return self._enrich_parent_context(user_id, ranked, analysis)

    def _vector_search(
        self,
        user_id: int,
        query_vector: list[float] | None,
        top_k: int,
        min_score: float,
        document_ids: list[int] | None,
    ) -> list[dict[str, Any]]:
        if not self.vector_store or not query_vector:
            return []
        try:
            hits = self.vector_store.search(
                user_id=user_id,
                query_vector=query_vector,
                top_k=max(top_k * 3, top_k),
                min_score=min_score,
                document_ids=document_ids,
            )
        except Exception as exc:
            logger.warning("vector_search_failed fallback_to_bm25 user_id=%s error=%s", user_id, exc, exc_info=True)
            return []
        normalized = _normalize_scores([float(hit.get("score", 0.0)) for hit in hits])
        for hit, score in zip(hits, normalized, strict=True):
            hit["vector_score"] = score
            hit.setdefault("bm25_score", 0.0)
            hit.setdefault("matched_terms", [])
            hit["retrieval_source"] = "vector"
            hit.setdefault("query_type", "semantic")
        return hits

    def _bm25_search(
        self,
        user_id: int,
        query: str,
        top_k: int,
        document_ids: list[int] | None,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        try:
            chunk_repo = DocumentChunkRepository(self.db)
            chunks = chunk_repo.list_child_candidates(user_id, document_ids=document_ids, limit=candidate_limit)
            if not chunks:
                return []
            docs_by_id = self._documents_by_id(user_id, [chunk.document_id for chunk in chunks])
            documents = []
            for chunk in chunks:
                metadata = chunk.metadata_json or {}
                chunk_id = str(metadata.get("chunk_id") or chunk.qdrant_point_id or chunk.id)
                documents.append(BM25Document(id=chunk_id, content=chunk.content, payload={"chunk": chunk, "document": docs_by_id.get(chunk.document_id)}))
            hits = bm25_search(query, documents, top_k=top_k)
        except Exception as exc:
            logger.warning("bm25_search_failed fallback_to_vector user_id=%s error=%s", user_id, exc, exc_info=True)
            return []

        results: list[dict[str, Any]] = []
        for hit in hits:
            chunk = hit.document.payload["chunk"]
            document = hit.document.payload.get("document")
            metadata = dict(chunk.metadata_json or {})
            chunk_id = str(metadata.get("chunk_id") or chunk.qdrant_point_id or chunk.id)
            results.append({
                "document_id": str(chunk.document_id),
                "chunk_id": chunk_id,
                "qdrant_point_id": chunk.qdrant_point_id,
                "chunk_index": chunk.chunk_index,
                "score": hit.normalized_score,
                "content_preview": chunk.content[:500],
                "content": chunk.content,
                "source_title": document.filename if document else "",
                "source_url": None,
                "filename": document.filename if document else "",
                "file_type": document.file_type if document else "",
                "token_count": chunk.token_count,
                "content_hash": metadata.get("content_hash", ""),
                "chunk_role": "child",
                "chunk_type": metadata.get("chunk_type", ""),
                "parent_id": metadata.get("parent_id"),
                "page_number": metadata.get("page_number"),
                "sheet_name": metadata.get("sheet_name"),
                "metadata": metadata,
                "heading_path": metadata.get("heading_path", []),
                "vector_score": 0.0,
                "bm25_score": hit.normalized_score,
                "bm25_raw_score": hit.score,
                "matched_terms": hit.matched_terms,
                "retrieval_source": "bm25",
            })
        return results

    def _merge_hits(self, vector_hits: list[dict[str, Any]], bm25_hits: list[dict[str, Any]], analysis: QueryAnalysis) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        order = 0
        for hit in vector_hits + bm25_hits:
            key = _hit_key(hit)
            existing = merged.get(key)
            if existing is None:
                item = dict(hit)
                item["_merge_order"] = order
                order += 1
                merged[key] = item
                continue
            existing["vector_score"] = max(float(existing.get("vector_score") or 0.0), float(hit.get("vector_score") or 0.0))
            existing["bm25_score"] = max(float(existing.get("bm25_score") or 0.0), float(hit.get("bm25_score") or 0.0))
            existing["bm25_raw_score"] = max(float(existing.get("bm25_raw_score") or 0.0), float(hit.get("bm25_raw_score") or 0.0))
            existing["matched_terms"] = _unique([*(existing.get("matched_terms") or []), *(hit.get("matched_terms") or [])])
            if existing["vector_score"] > 0 and existing["bm25_score"] > 0:
                existing["retrieval_source"] = "hybrid"
            elif existing["bm25_score"] > 0:
                existing["retrieval_source"] = "bm25"
            else:
                existing["retrieval_source"] = "vector"
            for field in ("content", "content_preview", "source_title", "filename", "file_type", "metadata", "heading_path", "parent_id"):
                if not existing.get(field) and hit.get(field):
                    existing[field] = hit[field]
        results = list(merged.values())
        for item in results:
            if item.get("vector_score", 0) > 0 and item.get("bm25_score", 0) > 0:
                item["retrieval_source"] = "hybrid"
            item["query_type"] = analysis.query_type
            item.setdefault("score", max(float(item.get("vector_score") or 0.0), float(item.get("bm25_score") or 0.0)))
        results.sort(key=lambda item: item.get("_merge_order", 0))
        for item in results:
            item.pop("_merge_order", None)
        return results

    def _enrich_parent_context(self, user_id: int, child_hits: list[dict[str, Any]], analysis: QueryAnalysis) -> list[dict[str, Any]]:
        parent_lookup_request: dict[int, list[str]] = defaultdict(list)
        for hit in child_hits:
            document_id = _as_int(hit.get("document_id"))
            parent_id = _parent_id_from_hit(hit)
            if document_id is not None and parent_id:
                parent_lookup_request[document_id].append(parent_id)

        parent_chunks = DocumentChunkRepository(self.db).list_parent_chunks(user_id, dict(parent_lookup_request))

        enriched: list[dict[str, Any]] = []
        for hit in child_hits:
            document_id = _as_int(hit.get("document_id"))
            parent_id = _parent_id_from_hit(hit)
            parent = parent_chunks.get((document_id, parent_id)) if document_id is not None and parent_id else None
            child_content = hit.get("content", "") or hit.get("content_preview", "")
            parent_metadata = parent.metadata_json if parent else {}
            parent_context = parent.content if parent else child_content
            enriched_hit = {
                **hit,
                "child_chunk_id": hit.get("chunk_id", ""),
                "parent_id": parent_id,
                "parent_chunk_id": parent_metadata.get("chunk_id") if parent else None,
                "parent_db_chunk_id": parent.id if parent else None,
                "parent_chunk_index": parent.chunk_index if parent else None,
                "parent_context": parent_context,
                "parent_context_available": parent is not None,
                "citation": {
                    "document_id": hit.get("document_id"),
                    "chunk_id": hit.get("chunk_id", ""),
                    "child_chunk_id": hit.get("chunk_id", ""),
                    "parent_id": parent_id,
                    "filename": hit.get("filename") or hit.get("source_title", ""),
                    "source_title": hit.get("source_title", ""),
                    "score": hit.get("score", 0.0),
                    "final_score": hit.get("final_score", hit.get("score", 0.0)),
                    "retrieval_source": hit.get("retrieval_source", "vector"),
                    "vector_score": hit.get("vector_score", 0.0),
                    "bm25_score": hit.get("bm25_score", 0.0),
                    "matched_terms": hit.get("matched_terms", []),
                    "query_type": analysis.query_type,
                    "heading_path": hit.get("heading_path") or (hit.get("metadata", {}) or {}).get("heading_path", []),
                    "page_number": hit.get("page_number") or (hit.get("metadata", {}) or {}).get("page_number"),
                    "sheet_name": hit.get("sheet_name") or (hit.get("metadata", {}) or {}).get("sheet_name"),
                },
            }
            enriched.append(enriched_hit)
        return enriched

    def _documents_by_id(self, user_id: int, document_ids: list[int]) -> dict[int, Document]:
        unique_ids = sorted({int(item) for item in document_ids if item})
        if not unique_ids:
            return {}
        stmt = select(Document).where(Document.user_id == user_id, Document.id.in_(unique_ids))
        return {document.id: document for document in self.db.execute(stmt).scalars()}


def retrieve(query: str, user_id: int) -> list[dict[str, Any]]:
    return []


def _parent_id_from_hit(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata", {}) or {}
    return str(hit.get("parent_id") or metadata.get("parent_id") or "")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hit_key(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata", {}) or {}
    return str(hit.get("child_chunk_id") or hit.get("chunk_id") or metadata.get("chunk_id") or hit.get("qdrant_point_id") or "")


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores) or 1.0
    return [max(0.0, min(1.0, score / max_score)) for score in scores]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
