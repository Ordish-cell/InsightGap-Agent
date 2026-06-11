from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.context.builder import ContextBuilder
from src.web_app.core.config import settings
from src.web_app.db.repositories.document_repository import DocumentRepository
from src.web_app.db.session import SessionLocal
from src.web_app.models.orm import Document, DocumentChunk
from src.web_app.rag.embeddings import embed_text
from src.web_app.rag.query_analyzer import analyze_query
from src.web_app.rag.retriever import ParentChildRetriever
from src.web_app.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def is_document_overview_query(query: str) -> bool:
    return analyze_query(query).is_summary


def is_general_knowledge_question(text: str) -> bool:
    lower = (text or "").lower()
    return any(pattern in lower for pattern in ["是什么", "解释", "介绍", "原理", "定义", "how", "what is", "why"])


def is_document_specific_question(text: str) -> bool:
    lower = (text or "").lower()
    return any(pattern in lower for pattern in ["文档", "文件", "附件", "上传", "知识库", "pdf", "document", "file"])


async def _answer_from_general_llm(question: str) -> str:
    from src.web_app.agent.llm.factory import get_chat_model

    try:
        model = get_chat_model("rag", complexity="normal", temperature=0.35)
        message = await model.ainvoke(f"请用简洁中文回答，不要编造信息。\n\n用户问题：{question}")
        content = getattr(message, "content", str(message))
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        return str(content)
    except Exception:
        logger.warning("general_llm_fallback_failed", exc_info=True)
        return "抱歉，我在尝试回答这个问题时遇到了技术问题。"


class RAGService:
    def dependency_status(self) -> dict[str, Any]:
        configured = bool(settings.qdrant_url)
        return {
            "qdrant_configured": configured,
            "collection": settings.qdrant_collection,
            "hybrid_backend": settings.rag_hybrid_backend,
            "hybrid_collection": settings.qdrant_hybrid_collection,
            "embedding_model": resolve_model_name("embedding").model,
            "answer_model": resolve_model_name("rag").model,
        }

    def ingest_text(self, user_id: int, document_id: int, text: str) -> dict[str, Any]:
        return {"status": "deprecated", "reason": "Use DocumentService.ingest_document for persisted RAG ingestion", "user_id": user_id, "document_id": document_id}

    def search(self, user_id: int, query: str, top_k: int = 5, min_score: float = 0.2, document_ids: list[int] | None = None, db: Session | None = None) -> dict[str, Any]:
        vector = None
        vector_store = None
        warning = None
        if settings.qdrant_url:
            try:
                vector = embed_text(query)
                vector_store = QdrantVectorStore()
            except Exception as exc:
                warning = f"Vector retrieval unavailable: {exc}"
                logger.warning("rag.vector_unavailable fallback_to_bm25 user_id=%s error=%s", user_id, exc, exc_info=True)
        else:
            warning = "Qdrant is not configured; using BM25 fallback"

        with _db_scope(db) as session:
            results = ParentChildRetriever(session, vector_store).search(
                user_id=user_id,
                query=query,
                query_vector=vector,
                top_k=top_k,
                min_score=min_score,
                document_ids=document_ids,
                backend=settings.rag_hybrid_backend,
                allow_fallback=settings.qdrant_hybrid_fallback,
            )

        response: dict[str, Any] = {"query": query, "results": results}
        result_warning = next((item.get("retrieval_warning") for item in results if item.get("retrieval_warning")), None)
        if result_warning:
            response["retrieval_warning"] = result_warning
        if warning and not results:
            response["warning"] = warning
        elif warning:
            response["retrieval_warning"] = warning
        return response

    def search_evidence(self, user_id: int, query: str, limit: int = 5, score_threshold: float = 0.3, document_ids: list[int] | None = None) -> list[dict[str, Any]]:
        try:
            search_result = self.search(user_id, query, top_k=limit, min_score=score_threshold, document_ids=document_ids)
        except Exception as exc:
            logger.warning("RAG search_evidence failed (non-blocking): %s", exc)
            return []
        results = search_result.get("results", [])
        return [
            {
                "id": item.get("chunk_id", ""),
                "content": (item.get("parent_context") or item.get("content", ""))[:800],
                "score": item.get("score", 0.0),
                "document_id": item.get("document_id", ""),
                "chunk_id": item.get("chunk_id", ""),
                "child_chunk_id": item.get("child_chunk_id") or item.get("chunk_id", ""),
                "parent_id": item.get("parent_id"),
                "parent_context": item.get("parent_context", ""),
                "source_name": item.get("source_title", ""),
                "source_url": item.get("source_url", ""),
                "metadata": item.get("metadata", {}),
            }
            for item in results
        ]

    def ask(self, user_id: int, question: str, top_k: int = 5, min_score: float = 0.2, document_ids: list[int] | None = None, answer_mode: str = "auto", db: Session | None = None) -> dict[str, Any]:
        search_result = self.search(user_id, question, top_k, min_score, document_ids, db=db)
        results = search_result.get("results", [])
        if not results:
            is_general = is_general_knowledge_question(question)
            is_doc_specific = is_document_specific_question(question)
            return {
                "answer": "没有找到足够证据来回答这个问题。",
                "answer_mode": "general_knowledge_fallback" if (is_general and not is_doc_specific) else "no_evidence",
                "evidence": [],
                "needs_general_fallback": is_general and not is_doc_specific,
                "context": {"gssc_used": True, "selected_chunks": 0, "token_estimate": 0, "embedding_model": resolve_model_name("embedding").model, "answer_model": resolve_model_name("rag").model},
            }

        evidence = self._evidence_from_results(results)
        context = ContextBuilder().build({"task": question, "evidence": evidence, "output_contract": "Answer only from evidence and cite chunk_id/source_title."})
        return {
            "answer": self._extractive_answer(question, evidence),
            "answer_mode": "extractive_fallback",
            "evidence": evidence,
            "context": {"gssc_used": True, "selected_chunks": len(evidence), "token_estimate": max(1, len(context) // 4), "embedding_model": resolve_model_name("embedding").model, "answer_model": resolve_model_name("rag").model},
        }

    def stats(self, db: Session, user_id: int) -> dict[str, Any]:
        collection = QdrantVectorStore().get_collection_stats() if settings.qdrant_url else {"collection": settings.qdrant_collection, "points_count": 0}
        documents_count = db.execute(select(func.count(Document.id)).where(Document.user_id == user_id)).scalar_one()
        chunks_count = db.execute(select(func.count(DocumentChunk.id)).where(DocumentChunk.user_id == user_id)).scalar_one()
        return {**collection, "documents_count": documents_count, "chunks_count": chunks_count}

    def ask_document(self, user_id: int, question: str, document_ids: list[int] | None = None, top_k: int = 8, overview_mode: bool = False, db: Session | None = None) -> dict[str, Any]:
        evidence: list[dict[str, Any]] = []
        if (overview_mode or is_document_overview_query(question)) and document_ids:
            with _db_scope(db) as session:
                repo = DocumentRepository(session)
                for document_id in document_ids:
                    document = repo.get_by_id_for_user(user_id, int(document_id))
                    if not document:
                        continue
                    metadata = document.metadata_json or {}
                    overview = metadata.get("overview") or {}
                    document_map = metadata.get("document_map") or {}
                    overview_text = overview.get("summary_text") or (_document_map_to_text(document_map) if document_map else "")
                    if overview_text:
                        evidence.append({
                            "document_id": str(document.id),
                            "chunk_id": "overview",
                            "child_chunk_id": "",
                            "parent_id": None,
                            "score": 1.0,
                            "chunk_index": 0,
                            "quote": overview_text[:1600],
                            "child_quote": "",
                            "source_title": document.filename,
                            "source_url": None,
                            "metadata": {"chunk_role": "overview", "document_map": document_map},
                            "overview": overview,
                            "document_map": document_map,
                        })

        search_result = self.search(user_id, question, top_k=top_k, min_score=0.1, document_ids=document_ids, db=db)
        results = search_result.get("results", [])
        if not results and not evidence:
            return {
                "answer": "没有从当前上传的文档中解析到足够正文内容。",
                "answer_mode": "no_evidence",
                "evidence": [],
                "context": {"gssc_used": True, "selected_chunks": 0, "token_estimate": 0, "embedding_model": resolve_model_name("embedding").model, "answer_model": resolve_model_name("rag").model},
            }

        evidence.extend(self._evidence_from_results(results, existing=evidence))
        context_text = self._document_context_block(evidence)
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

    def _evidence_from_results(self, results: list[dict[str, Any]], existing: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen = {(str(item.get("document_id", "")), str(item.get("parent_id") or item.get("chunk_id") or "")) for item in existing or []}
        for item in results:
            parent_id = str(item.get("parent_id") or "")
            child_chunk_id = str(item.get("child_chunk_id") or item.get("chunk_id") or "")
            key = (str(item.get("document_id", "")), parent_id or child_chunk_id)
            if key in seen:
                continue
            seen.add(key)
            evidence.append({
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "child_chunk_id": child_chunk_id,
                "parent_id": item.get("parent_id"),
                "parent_chunk_id": item.get("parent_chunk_id"),
                "parent_context_available": item.get("parent_context_available", False),
                "score": item["score"],
                "quote": (item.get("parent_context") or item["content"])[:1200],
                "child_quote": item["content"][:800],
                "source_title": item["source_title"],
                "source_url": item["source_url"],
                "citation": item.get("citation", {}),
                "metadata": item["metadata"],
            })
        return evidence

    def _document_context_block(self, evidence: list[dict[str, Any]]) -> str:
        doc_names = {str(item.get("source_title", "document")) for item in evidence}
        parts = [f"[Document QA context: {', '.join(sorted(doc_names))}]"]
        for index, item in enumerate(evidence[:12], 1):
            parts.append(f"Chunk {index}:\n{item.get('quote', '')}")
        return "\n\n".join(parts)

    def _extractive_answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        lines = []
        for item in evidence[:8]:
            if item.get("quote"):
                lines.append(f"[{item.get('source_title', 'document')} #{item.get('chunk_id', '')}] {item['quote'].strip()[:800]}")
        return "以下是从当前上传文档中检索到的相关内容，请基于这些内容回答：\n\n" + "\n\n".join(lines)


@contextmanager
def _db_scope(db: Session | None):
    if db is not None:
        yield db
        return
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _document_map_to_text(document_map: dict[str, Any]) -> str:
    lines = [f"File: {document_map.get('filename', '')}", f"Type: {document_map.get('file_type', '')}"]
    sections = document_map.get("sections") or []
    if sections:
        lines.append("Sections:")
        for section in sections[:20]:
            title = " > ".join(section.get("heading_path") or []) or section.get("chunk_type", "section")
            if section.get("page_number"):
                title += f" page={section.get('page_number')}"
            elif section.get("sheet_name"):
                title += f" sheet={section.get('sheet_name')} rows={section.get('row_start')}-{section.get('row_end')}"
            lines.append(f"- {title}")
    return "\n".join(line for line in lines if line)


rag_service = RAGService()
