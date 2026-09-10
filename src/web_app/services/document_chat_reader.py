"""Bounded, read-only document access. Called in a worker with its own session."""
from sqlalchemy import select
from src.web_app.db.session import SessionLocal
from src.web_app.models.orm import Document, DocumentChunk
from src.web_app.services.conversation_files import conversation_files


def read_documents(user_id, conversation_id, decision, token_budget=8000):
    with SessionLocal() as db:
        return read_documents_in_session(db, user_id, conversation_id, decision, token_budget)


def read_documents_in_session(db, user_id, conversation_id, decision, token_budget=8000):
    inventory = {f["document_id"]: f for f in conversation_files(db, user_id, conversation_id)}
    ids = decision["document_ids"]
    if not ids or len(ids) > 3 or any(i not in inventory for i in ids):
        raise ValueError("document_scope_invalid")
    results = []
    # A character costs at most a few tokens; use a conservative UTF-8 byte budget.
    remaining = max(0, token_budget)
    for index, document_id in enumerate(ids):
        allocation = remaining // (len(ids) - index)
        info = inventory[document_id]
        item = {**info, "text": "", "references": [], "coverage": "none"}
        results.append(item)
        if info["status"] not in {"ingested", "ready", "completed"}:
            continue
        doc = db.scalar(select(Document).where(Document.id == document_id, Document.user_id == user_id))
        base = select(DocumentChunk).where(DocumentChunk.document_id == document_id, DocumentChunk.user_id == user_id)
        parent_role = DocumentChunk.metadata_json["chunk_role"].as_string() == "parent"
        has_parents = db.scalar(select(DocumentChunk.id).where(DocumentChunk.document_id == document_id,
            DocumentChunk.user_id == user_id, parent_role).limit(1)) is not None
        query = base.where(parent_role) if has_parents else base
        chunks, parts, size = [], [], 0
        for chunk in db.scalars(query.order_by(DocumentChunk.chunk_index).execution_options(yield_per=100)):
            if not has_parents and (chunk.metadata_json or {}).get("chunk_role") in {"overview", "summary"}:
                continue
            chunks.append(chunk)
            parts.append(chunk.content)
            size += len(chunk.content.encode("utf-8")) + 2
            if size > allocation + 2:
                break
        text = "\n\n".join(parts)
        full_size = len(text.encode("utf-8"))
        if text and full_size <= allocation:
            item.update(text=text, coverage="full", references=[{"document_id": document_id, "chunk_id": c.id} for c in chunks])
        elif decision.get("mode") == "overview":
            meta = doc.metadata_json or {}
            overview = meta.get("overview") or {}
            summary = overview.get("summary_text") if overview.get("summary_status") == "generated" else ""
            structure = (meta.get("document_map") or {}).get("sections") or []
            headings = "\n".join(" > ".join(s.get("heading_path") or []) for s in structure[:20] if isinstance(s, dict))
            item.update(text=(str(summary) + "\n目录：\n" + headings) if summary else text,
                coverage="summary" if summary else "partial", references=[{"document_id": document_id,
                "chunk_id": "overview"}] if summary else
                [{"document_id": document_id, "chunk_id": c.id} for c in chunks])
        else:
            try:
                from src.web_app.services.rag_service import rag_service
                found = rag_service.search(user_id, decision["query"], top_k=5, document_ids=[document_id], db=db)
                evidence = rag_service._evidence_from_results(found.get("results", []))
                item.update(text="\n\n".join(e["quote"] for e in evidence), coverage="retrieved", references=[
                    {k: e[k] for k in ("document_id", "chunk_id", "source_title")} for e in evidence])
            except Exception:
                item.update(status="retrieval_failed")
        raw = item["text"].encode("utf-8")
        if len(raw) > allocation:
            item["text"] = raw[:allocation].decode("utf-8", errors="ignore")
            item["coverage"] = "partial"
        remaining -= len(item["text"].encode("utf-8"))
    return results
