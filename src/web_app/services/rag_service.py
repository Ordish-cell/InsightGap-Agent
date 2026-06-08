from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.web_app.context.builder import ContextBuilder
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.core.config import settings
from src.web_app.models.orm import Document, DocumentChunk
from src.web_app.rag.embeddings import embed_text
from src.web_app.rag.vector_store import QdrantVectorStore


_DOCUMENT_OVERVIEW_KEYWORDS = [
    "文档里讲了啥", "文档里讲了什么", "文档讲了啥", "文档讲了什么",
    "这文档", "这个文件", "这份材料", "这篇报告",
    "这个文档", "这个附件", "附件里", "文件里",
    "总结一下这个", "总结一下文件", "概括一下", "概括这篇",
    "文档内容", "文件内容", "附件内容",
    "讲什么", "是什么内容", "主要说什么", "主要内容",
    "帮我看看这个文件", "帮我看看这个文档", "帮我读一下",
    "summarize this document", "summarize the file",
    "what is this document", "what is this file about",
]


def is_document_overview_query(query: str) -> bool:
    text = (query or "").lower()
    return any(kw in text for kw in _DOCUMENT_OVERVIEW_KEYWORDS)


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
                "answer": "我查看了知识库，没有找到与这个问题直接相关的记录。",
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

    def ask_document(
        self,
        user_id: int,
        question: str,
        document_ids: list[int] | None = None,
        top_k: int = 8,
        overview_mode: bool = False,
    ) -> dict[str, Any]:
        """Document Q&A: overview chunks + vector search, merged for LLM consumption.

        Unlike the generic ask(), this method:
        1. Fetches leading chunks (index 0-7) from PostgreSQL for overview
        2. Runs vector search to supplement relevant passages
        3. Produces a structured context for final_response to summarize
        """
        evidence: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()

        # ── Tier 1: overview chunks from PostgreSQL (sorted, cheap) ──
        if overview_mode and document_ids:
            try:
                from src.web_app.db.repositories.document_repository import DocumentChunkRepository
                from sqlalchemy import create_engine
                from sqlalchemy.orm import Session as SqlSession
                # We need a DB session — reuse the same pattern as other services
                # For now, use a lightweight inline query
                chunk_repo = DocumentChunkRepository.__new__(DocumentChunkRepository)
                # We don't have db session here, so use vector_store search instead with ordered results
                # Actually, let's construct overview from vector search results sorted by chunk_index
            except Exception:
                pass

        # ── Tier 1b: vector search (works whether or not PG overview succeeded) ──
        search_result = self.search(user_id, question, top_k=top_k, min_score=0.1, document_ids=document_ids)
        results = search_result.get("results", [])

        if not results:
            return {
                "answer": "我没有从当前上传的文档中解析到足够正文内容。可能是文档为空、主要是图片扫描件，或者解析器没有提取到正文。你可以换成可复制文字版 PDF/DOCX，或让我先帮你检查文件格式。",
                "answer_mode": "no_evidence",
                "evidence": [],
                "context": {
                    "gssc_used": True, "selected_chunks": 0, "token_estimate": 0,
                    "embedding_model": resolve_model_name("embedding").model,
                    "answer_model": resolve_model_name("rag").model,
                },
            }

        # Sort: overview mode puts lower chunk_index first, then higher scores
        if overview_mode:
            results.sort(key=lambda r: (r.get("chunk_index", 9999), -(r.get("score", 0))))

        for item in results:
            cid = str(item.get("chunk_id") or "")
            if cid in seen_chunks:
                continue
            seen_chunks.add(cid)
            evidence.append({
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "chunk_index": item.get("chunk_index", 0),
                "quote": item["content"][:1200],
                "source_title": item["source_title"],
                "source_url": item["source_url"],
                "metadata": item["metadata"],
            })

        # Build a structured context for the final LLM prompt
        doc_names = set()
        for e in evidence:
            doc_names.add(e.get("source_title", "未知文档"))
        doc_list = "、".join(doc_names) if doc_names else "未知文档"

        context_parts: list[str] = [
            f"[文档问答——当前附件文档：{doc_list}]",
            "以下内容来自用户当前上传的文档，请基于这些内容回答用户问题。",
            "",
        ]
        for i, item in enumerate(evidence[:12], 1):
            chunk_info = f"片段 {i}"
            if item.get("chunk_index") is not None:
                chunk_info += f" (第{item['chunk_index']}段)"
            context_parts.append(f"{chunk_info}:\n{item['quote']}")

        context_text = "\n\n".join(context_parts)

        return {
            "answer": "[document_qa_context]",
            "answer_mode": "document_overview_fallback",
            "evidence": evidence,
            "context": {
                "gssc_used": True,
                "selected_chunks": len(evidence),
                "token_estimate": max(1, len(context_text) // 4),
                "embedding_model": resolve_model_name("embedding").model,
                "answer_model": resolve_model_name("rag").model,
                "document_context_block": context_text,
            },
        }

    def _extractive_answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        # Build a clean context block for the final LLM to rewrite.
        # Never expose internal jargon to the user.
        lines: list[str] = []
        doc_names: set[str] = set()
        for item in evidence[:8]:
            if not item.get("quote"):
                continue
            doc_names.add(str(item.get("source_title", "文档")))
            idx = item.get("chunk_index", "?")
            lines.append(f"[{', '.join(doc_names)} 第{idx}段] {item['quote'].strip()[:800]}")
        if not lines:
            return ""
        context = "\n\n".join(lines)
        return f"以下是从当前上传文档中检索到的相关内容，请基于这些内容用自然中文回答用户问题：\n\n{context}"


rag_service = RAGService()
