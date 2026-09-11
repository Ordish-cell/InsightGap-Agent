"""Versioned index-only text. Raw text and citations remain unchanged."""
from src.web_app.core.config import settings
from src.web_app.rag.embeddings import MAX_EMBED_CHARS


def retrieval_text(chunk, filename="", document_metadata=None, *, mode=None):
    content = chunk["content"]
    mode = mode or settings.rag_index_context_mode
    if mode == "raw":
        return content
    if mode != "metadata":
        raise ValueError("Unsupported RAG index context mode")
    meta = {**(document_metadata or {}), **(chunk.get("metadata") or {})}
    lines = [f"File: {filename}", "Section: " + " > ".join(chunk.get("heading_path") or meta.get("heading_path") or [])]
    for key in ("version", "effective_date", "sheet_name"):
        if meta.get(key) is not None:
            lines.append(f"{key}: {meta[key]}")
    prefix = "\n".join(lines) + "\n\n"
    # Never drop raw content to make room for metadata.
    space = max(0, MAX_EMBED_CHARS - len(content))
    return prefix[:min(700, space)] + content


def indexed_chunk(chunk, filename="", document_metadata=None):
    return {**chunk, "retrieval_text": retrieval_text(chunk, filename, document_metadata),
            "metadata": {**(chunk.get("metadata") or {}),
                         "index_context_version": "metadata-v1" if settings.rag_index_context_mode == "metadata" else "raw-v1"}}
