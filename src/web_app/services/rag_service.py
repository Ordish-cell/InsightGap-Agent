from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.web_app.context.builder import ContextBuilder
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.core.config import settings
from src.web_app.models.orm import Document, DocumentChunk
from src.web_app.rag.embeddings import embed_text
from src.web_app.rag.vector_store import QdrantVectorStore


class RAGService:
    def dependency_status(self) -> dict[str, Any]:
        configured = bool(settings.qdrant_url)
        return {"qdrant_configured": configured, "collection": settings.qdrant_collection, "embedding_model": resolve_model_name("embedding").model, "answer_model": resolve_model_name("rag").model}

    def ingest_text(self, user_id: int, document_id: int, text: str) -> dict[str, Any]:
        return {"status": "deprecated", "reason": "Use DocumentService.ingest_document for persisted RAG ingestion", "user_id": user_id, "document_id": document_id}

    def search(self, user_id: int, query: str, top_k: int = 5, min_score: float = 0.2, document_ids: list[int] | None = None) -> dict[str, Any]:
        if not settings.qdrant_url:
            return {"query": query, "results": [], "error": "Qdrant is not configured"}
        vector = embed_text(query)
        results = QdrantVectorStore().search(user_id=user_id, query_vector=vector, top_k=top_k, min_score=min_score, document_ids=document_ids)
        return {"query": query, "results": results}

    def search_evidence(
        self,
        user_id: int,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.3,
        document_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Lightweight RAG evidence retrieval — no LLM call.

        Used by context_builder to inject evidence into GSSC Gather phase.
        Returns a list of evidence dicts suitable for ContextBuilder packets.
        Qdrant unavailability is a non-blocking warning.
        """
        if not settings.qdrant_url:
            return []
        import logging
        _logger = logging.getLogger("rag_service")
        try:
            search_result = self.search(user_id, query, top_k=limit, min_score=score_threshold, document_ids=document_ids)
        except Exception as exc:
            _logger.warning("RAG search_evidence failed (non-blocking): %s", exc)
            return []
        results = search_result.get("results", [])
        if not results:
            return []
        return [
            {
                "id": item.get("chunk_id", ""),
                "content": item.get("content", "")[:800],
                "score": item.get("score", 0.0),
                "document_id": item.get("document_id", ""),
                "chunk_id": item.get("chunk_id", ""),
                "source_name": item.get("source_title", ""),
                "source_url": item.get("source_url", ""),
                "metadata": item.get("metadata", {}),
            }
            for item in results
        ]

    def ask(self, user_id: int, question: str, top_k: int = 5, min_score: float = 0.2, document_ids: list[int] | None = None, answer_mode: str = "auto") -> dict[str, Any]:
        search_result = self.search(user_id, question, top_k, min_score, document_ids)
        results = search_result.get("results", [])
        if not results:
            return {
                "answer": "当前知识库中没有找到足够证据回答该问题。",
                "answer_mode": "no_evidence",
                "evidence": [],
                "context": {"gssc_used": True, "selected_chunks": 0, "token_estimate": 0, "embedding_model": resolve_model_name("embedding").model, "answer_model": resolve_model_name("rag").model},
            }
        evidence = [
            {
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "quote": item["content"][:800],
                "source_title": item["source_title"],
                "source_url": item["source_url"],
                "metadata": item["metadata"],
            }
            for item in results
        ]
        context = ContextBuilder().build({"task": question, "evidence": evidence, "output_contract": "Answer only from evidence and cite chunk_id/source_title."})
        answer = self._extractive_answer(question, evidence)
        return {
            "answer": answer,
            "answer_mode": "extractive_fallback",
            "evidence": evidence,
            "context": {"gssc_used": True, "selected_chunks": len(evidence), "token_estimate": max(1, len(context) // 4), "embedding_model": resolve_model_name("embedding").model, "answer_model": resolve_model_name("rag").model},
        }

    def stats(self, db: Session, user_id: int) -> dict[str, Any]:
        collection = QdrantVectorStore().get_collection_stats() if settings.qdrant_url else {"collection": settings.qdrant_collection, "points_count": 0}
        documents_count = db.execute(select(func.count(Document.id)).where(Document.user_id == user_id)).scalar_one()
        chunks_count = db.execute(select(func.count(DocumentChunk.id)).where(DocumentChunk.user_id == user_id)).scalar_one()
        return {**collection, "documents_count": documents_count, "chunks_count": chunks_count}

    def _extractive_answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        quotes = [item["quote"].strip().replace("\n", " ") for item in evidence[:3] if item.get("quote")]
        joined = " ".join(quotes)
        if len(joined) > 700:
            joined = joined[:700] + "..."
        return f"基于当前知识库证据，问题“{question}”的相关信息是：{joined}"


rag_service = RAGService()
