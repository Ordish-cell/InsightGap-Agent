"""Resume index-only rebuilding into a NEW collection; never rewrites source chunks.

Default is dry-run. --apply writes only the named target; --checkpoint identifies
ownership on resume. Repeat under an ingestion pause before switching settings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fingerprint(document, chunks):
    value = {"filename": document.filename, "metadata": document.metadata_json,
             "chunks": [{"content": c.content, "metadata": c.metadata_json, "index": c.chunk_index} for c in chunks]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_parents(chunks):
    parents = {(c.metadata_json or {}).get("chunk_id") for c in chunks if (c.metadata_json or {}).get("chunk_role") == "parent"}
    for chunk in chunks:
        meta = chunk.metadata_json or {}
        if meta.get("chunk_role") == "child" and meta.get("parent_id") not in parents:
            raise ValueError("Missing parent for persisted child")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--eval-manifest", type=Path, help="Export a metadata-index copy of this isolated evaluation manifest")
    args = parser.parse_args()
    from sqlalchemy import select
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.web_app.core.config import settings
    from src.web_app.db.session import SessionLocal
    from src.web_app.models.orm import Document, DocumentChunk
    from src.web_app.rag.vector_store import QdrantVectorStore
    from src.web_app.rag.index_text import indexed_chunk
    from src.web_app.rag.embeddings import embed_texts
    source = settings.qdrant_hybrid_collection
    if args.target in {source, settings.qdrant_collection, settings.memory_qdrant_collection}:
        parser.error("Target must be a new collection, never source or memory")
    if not args.target.startswith(("rag_eval_", "agent_os_documents_")):
        parser.error("Target must use rag_eval_ or agent_os_documents_ prefix")
    settings.rag_hybrid_backend = "qdrant_hybrid"
    settings.rag_index_context_mode = "metadata"
    settings.qdrant_hybrid_collection = args.target
    store = QdrantVectorStore()
    config = {"source": source, "target": args.target, "user_id": args.user_id, "embedding": settings.embed_model_name,
              "vector_size": settings.qdrant_vector_size, "sparse": settings.qdrant_sparse_encoder,
              "sparse_hash_size": settings.qdrant_sparse_hash_size, "index_version": "metadata-v1"}
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else {"config": config, "documents": {}}
    if checkpoint["config"] != config:
        parser.error("Checkpoint configuration differs from this run")
    exists = args.target in {c.name for c in store.client.get_collections().collections}
    if exists and not args.checkpoint.exists():
        parser.error("Existing target has no ownership checkpoint")
    def persist():
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.checkpoint)
    if args.apply:
        checkpoint["complete"] = False
        persist()
        store.ensure_collection()
    with SessionLocal() as db:
        query = select(Document).where(Document.status.in_(["ingested", "ready", "completed"]))
        if args.user_id is not None:
            query = query.where(Document.user_id == args.user_id)
        documents = list(db.scalars(query.order_by(Document.id)))
        current = {str(d.id) for d in documents}
        for doc_id, saved in list(checkpoint["documents"].items()):
            if doc_id not in current:
                print(f"target stale document: {doc_id}")
                if args.apply:
                    store.delete_document(saved["user_id"], int(doc_id))
                    del checkpoint["documents"][doc_id]
                    persist()
        for doc in documents:
            chunks = list(db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id, DocumentChunk.user_id == doc.user_id).order_by(DocumentChunk.chunk_index)))
            validate_parents(chunks)
            digest = fingerprint(doc, chunks)
            saved = checkpoint["documents"].get(str(doc.id))
            if saved and saved["hash"] == digest and exists:
                count_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=str(doc.user_id))), FieldCondition(key="document_id", match=MatchValue(value=str(doc.id)))])
                if store.client.count(args.target, count_filter=count_filter, exact=True).count == saved["points"]:
                    continue
            vectors = [indexed_chunk({"content": c.content, "chunk_index": c.chunk_index, "token_count": c.token_count,
                "metadata": c.metadata_json or {}, "heading_path": (c.metadata_json or {}).get("heading_path", []),
                "point_id": str(uuid5(NAMESPACE_URL, f"{args.target}:{doc.user_id}:{doc.id}:{(c.metadata_json or {}).get('chunk_id', c.id)}"))}, doc.filename, doc.metadata_json)
                for c in chunks if (c.metadata_json or {}).get("chunk_role") in {"child", "overview", "section_summary"}]
            print(f"rebuild document={doc.id} points={len(vectors)} apply={args.apply}")
            if not args.apply:
                continue
            store.delete_document(doc.user_id, doc.id)
            for start in range(0, len(vectors), 10):
                batch = vectors[start:start + 10]
                store.upsert_chunks(doc.user_id, doc.id, batch, embed_texts([c["retrieval_text"] for c in batch]), doc)
            count_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=str(doc.user_id))), FieldCondition(key="document_id", match=MatchValue(value=str(doc.id)))])
            if store.client.count(args.target, count_filter=count_filter, exact=True).count != len(vectors):
                raise RuntimeError("Target point count mismatch")
            db.expire_all()
            fresh = list(db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id, DocumentChunk.user_id == doc.user_id).order_by(DocumentChunk.chunk_index)))
            if fingerprint(doc, fresh) != digest or doc.status not in {"ingested", "ready", "completed"}:
                raise RuntimeError("Source changed during rebuild; resume after pausing ingestion")
            checkpoint["documents"][str(doc.id)] = {"hash": digest, "points": len(vectors), "user_id": doc.user_id}
            persist()
    if args.apply:
        checkpoint["complete"] = True
        checkpoint["cutover_ready"] = False  # Requires paused final delta + retrieval/quality validation.
        persist()
        if args.eval_manifest:
            original = json.loads(args.eval_manifest.read_text(encoding="utf-8"))
            if not args.target.startswith("rag_eval_") or original.get("user_id") != args.user_id or not original.get("complete"):
                raise ValueError("Evaluation manifest ownership mismatch")
            if {str(d["id"]) for d in original["documents"]} != set(checkpoint["documents"]):
                raise ValueError("Evaluation manifest document set mismatch")
            target_manifest = args.eval_manifest.with_name(args.eval_manifest.stem + "_metadata.json")
            target_manifest.write_text(json.dumps({**original, "collection": args.target, "index_version": "metadata"}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
