import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.web_app.core.config import settings
from src.web_app.core.errors import DocumentIngestError
from src.web_app.db.repositories.document_repository import DocumentChunkRepository, DocumentRepository
from src.web_app.models.orm import Document
from src.web_app.rag.document_parser import ALLOWED_EXTENSIONS, parse_document
from src.web_app.rag.embeddings import MAX_EMBED_CHARS, embed_texts
from src.web_app.rag.structured_chunker import build_structured_chunks, fallback_structured_chunks
from src.web_app.rag.vector_store import QdrantVectorStore

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CHAT_UPLOAD_BYTES = getattr(settings, "max_chat_upload_bytes", None) or 20 * 1024 * 1024

ALLOWED_CHAT_UPLOAD_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".json", ".html", ".htm",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

logger = logging.getLogger(__name__)

INGESTED_STATUSES = {"ingested", "completed", "ready"}


def document_to_dict(document) -> dict[str, Any]:
    return {
        "id": document.id,
        "user_id": document.user_id,
        "filename": document.filename,
        "file_path": document.file_path,
        "file_type": document.file_type,
        "source_type": document.source_type,
        "status": document.status,
        "metadata": document.metadata_json or {},
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "updated_at": document.updated_at.isoformat() if document.updated_at else None,
    }


class DocumentService:
    def _delete_document_vectors(self, user_id: int, document_id: int, *, required: bool) -> str | None:
        """Delete only document RAG vectors for a document.

        Returns a warning string when cleanup is skipped or fails in a best-effort
        path. If required=True, configured Qdrant delete failures are raised so
        ingest cannot drift further out of sync.
        """
        if not settings.qdrant_url:
            warning = "Qdrant is not configured; document vectors were not cleaned"
            logger.warning("document.vector_cleanup_skipped user_id=%s document_id=%s reason=%s", user_id, document_id, warning)
            return warning
        try:
            QdrantVectorStore().delete_document(user_id, document_id)
            return None
        except Exception as exc:
            warning = f"Document vector cleanup failed: {exc}"
            logger.warning("document.vector_cleanup_failed user_id=%s document_id=%s error=%s", user_id, document_id, exc, exc_info=True)
            if required:
                raise RuntimeError(warning) from exc
            return warning

    def upload_document(self, db: Session, user_id: int, file: UploadFile) -> dict[str, Any]:
        filename = Path(file.filename or "").name
        if not filename:
            raise ValueError("Filename is required")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        repo = DocumentRepository(db)
        document = repo.create_uploaded_document(user_id=user_id, filename=filename, file_path="", file_type=suffix.lstrip("."), metadata={"original_filename": filename})
        base_dir = self._document_dir(user_id, document.id)
        base_dir.mkdir(parents=True, exist_ok=True)
        target = (base_dir / filename).resolve()
        if not str(target).startswith(str(base_dir.resolve())):
            raise ValueError("Invalid upload path")

        size = 0
        with target.open("wb") as handle:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    repo.mark_failed(document, "File is too large", failed_stage="upload")
                    target.unlink(missing_ok=True)
                    raise ValueError("File is too large")
                handle.write(chunk)
        repo.update(document, file_path=str(target), metadata_json={**(document.metadata_json or {}), "size": size})
        return document_to_dict(document)

    def ingest_document(self, db: Session, user_id: int, document_id: int) -> dict[str, Any]:
        doc_repo = DocumentRepository(db)
        chunk_repo = DocumentChunkRepository(db)
        document = doc_repo.get_by_id_for_user(user_id, document_id)
        if not document:
            raise ValueError("Document not found")
        try:
            doc_repo.update_status(document, "ingesting", {"failed_stage": None, "error": None, "error_message": None})
            doc_repo.update_status(document, "ingesting", {"failed_stage": "qdrant_delete"})
            self._delete_document_vectors(user_id, document.id, required=True)
            chunk_repo.delete_by_document(user_id, document.id)
            ingest_result = self._ingest_document_internal(db, user_id, document, cleanup_existing=False)
            token_count = ingest_result["token_count"]
            return {"document": document_to_dict(document), "chunk_count": ingest_result["chunk_count"], "token_count": token_count}
        except Exception as exc:
            failed_stage = (document.metadata_json or {}).get("failed_stage") or "ingest"
            doc_repo.mark_failed(document, str(exc), failed_stage=failed_stage)
            raise

    def list_documents(self, db: Session, user_id: int) -> list[dict[str, Any]]:
        return [document_to_dict(item) for item in DocumentRepository(db).list_by_user(user_id)]

    def get_document(self, db: Session, user_id: int, document_id: int) -> dict[str, Any]:
        document = DocumentRepository(db).get_by_id_for_user(user_id, document_id)
        if not document:
            raise ValueError("Document not found")
        chunks = DocumentChunkRepository(db).list_by_document(user_id, document_id)
        data = document_to_dict(document)
        data["chunks"] = [{"id": chunk.id, "chunk_index": chunk.chunk_index, "token_count": chunk.token_count, "metadata": chunk.metadata_json or {}} for chunk in chunks]
        return data

    def delete_document(self, db: Session, user_id: int, document_id: int) -> dict[str, Any]:
        document = DocumentRepository(db).get_by_id_for_user(user_id, document_id)
        if not document:
            raise ValueError("Document not found")
        vector_cleanup_warning = self._delete_document_vectors(user_id, document_id, required=False)
        DocumentChunkRepository(db).delete_by_document(user_id, document_id)
        file_cleanup_warning = None
        path = Path(document.file_path)
        if path.exists():
            try:
                shutil.rmtree(path.parent)
            except Exception as exc:
                file_cleanup_warning = f"Local file cleanup failed: {exc}"
                logger.warning("document.file_cleanup_failed user_id=%s document_id=%s path=%s error=%s", user_id, document_id, path.parent, exc, exc_info=True)
        db.delete(document)
        db.commit()
        result = {"deleted": True, "document_id": document_id}
        if vector_cleanup_warning:
            result["vector_cleanup_warning"] = vector_cleanup_warning
        if file_cleanup_warning:
            result["file_cleanup_warning"] = file_cleanup_warning
        return result

    def upload_chat_attachment(self, db: Session, user_id: int, file: UploadFile) -> dict[str, Any]:
        filename = Path(file.filename or "").name
        if not filename:
            raise ValueError("Filename is required")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_CHAT_UPLOAD_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        is_image = suffix in IMAGE_EXTENSIONS
        kind = "image" if is_image else "document"

        repo = DocumentRepository(db)
        stored_name = f"{uuid4().hex}{suffix}"
        document = repo.create(
            user_id=user_id,
            filename=filename,
            file_path="",
            file_type=suffix.lstrip("."),
            source_type="chat_upload",
            status="uploaded",
            metadata_json={
                "upload_type": "chat",
                "original_filename": filename,
                "stored_filename": stored_name,
                "mime_type": file.content_type or "",
                "kind": kind,
                "ingest_status": "pending",
                "chunk_count": 0,
                "error": None,
            },
        )

        base_dir = self._document_dir(user_id, document.id)
        base_dir.mkdir(parents=True, exist_ok=True)
        target = (base_dir / stored_name).resolve()
        if not str(target).startswith(str(base_dir.resolve())):
            raise ValueError("Invalid upload path")

        size = 0
        with target.open("wb") as handle:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_CHAT_UPLOAD_BYTES:
                    repo.mark_failed(document, "File is too large", failed_stage="upload")
                    target.unlink(missing_ok=True)
                    raise ValueError("File is too large")
                handle.write(chunk)

        metadata = dict(document.metadata_json or {})
        metadata["size"] = size
        repo.update(document, file_path=str(target), metadata_json=metadata)

        # ── Sync ingest for documents (images skip) ──────────────────────
        if kind == "document":
            try:
                ingest_result = self._ingest_document_internal(db, user_id, document)
                if ingest_result.get("status") != "ingested":
                    error_msg = str(ingest_result.get("error") or "Document ingestion returned non-ingested status")
                    repo.mark_failed(document, error_msg, failed_stage="ingest")
                    repo.update(document, metadata_json={
                        **(document.metadata_json or {}),
                        "ingest_status": "failed",
                        "error": error_msg,
                        "ingest_error_type": "IngestError",
                    })
                    raise DocumentIngestError(
                        f"文档摄入失败：{error_msg}",
                        document_id=document.id,
                        http_status=500,
                        detail=error_msg,
                    )
                metadata = dict(document.metadata_json or {})
                metadata["ingest_status"] = "ingested"
                metadata["chunk_count"] = ingest_result.get("chunk_count", 0)
                metadata["token_count"] = ingest_result.get("token_count", 0)
                repo.update(document, status="ingested", metadata_json=metadata)
                return {
                    "document_id": document.id,
                    "filename": filename,
                    "file_type": suffix.lstrip("."),
                    "mime_type": file.content_type or "",
                    "kind": kind,
                    "size": size,
                    "preview_url": f"/api/v1/documents/{document.id}/file",
                    "status": "ready",
                    "ingest_status": "ingested",
                    "chunks_count": ingest_result.get("chunk_count", 0),
                    "token_count": ingest_result.get("token_count", 0),
                }
            except DocumentIngestError:
                raise
            except Exception as exc:
                logger.exception("Chat document sync ingest failed for document_id=%s", document.id)
                failed_stage = (document.metadata_json or {}).get("failed_stage") or "ingest"
                repo.mark_failed(document, str(exc), failed_stage=failed_stage)
                repo.update(document, metadata_json={
                    **(document.metadata_json or {}),
                    "ingest_status": "failed",
                    "error": str(exc),
                    "ingest_error_type": type(exc).__name__,
                })
                http_status = 400 if isinstance(exc, (ValueError, FileNotFoundError)) else 500
                raise DocumentIngestError(
                    str(exc),
                    document_id=document.id,
                    http_status=http_status,
                    detail=str(exc),
                ) from exc

        return {
            "document_id": document.id,
            "filename": filename,
            "file_type": suffix.lstrip("."),
            "mime_type": file.content_type or "",
            "kind": kind,
            "size": size,
            "preview_url": f"/api/v1/documents/{document.id}/file",
            "status": "ready",
            "ingest_status": "completed",
            "chunks_count": 0,
            "token_count": 0,
        }

    def _ingest_document_internal(self, db: Session, user_id: int, document: Document, *, cleanup_existing: bool = True) -> dict[str, Any]:
        """Sync parse + chunk + embed + upsert. Returns result dict, raises on error."""
        doc_repo = DocumentRepository(db)
        chunk_repo = DocumentChunkRepository(db)
        if cleanup_existing:
            doc_repo.update_status(document, "ingesting", {"failed_stage": None, "error": None, "error_message": None})
            doc_repo.update_status(document, "ingesting", {"failed_stage": "qdrant_delete"})
            self._delete_document_vectors(user_id, document.id, required=True)
            chunk_repo.delete_by_document(user_id, document.id)
        try:
            doc_repo.update_status(document, "ingesting", {"failed_stage": "parse"})
            parsed = parse_document(document.file_path, document.filename, document.file_type)
            doc_repo.update_status(document, "ingesting", {"failed_stage": "chunk"})
            try:
                structured = build_structured_chunks(
                    parsed["markdown"] or parsed["text"],
                    file_type=document.file_type,
                    filename=document.filename,
                    parser_metadata=parsed.get("metadata", {}),
                )
            except Exception as exc:
                logger.warning("document.structured_chunk_failed document_id=%s error=%s", document.id, exc, exc_info=True)
                structured = fallback_structured_chunks(parsed["markdown"] or parsed["text"])
                structured["stats"] = {**structured.get("stats", {}), "structured_error": str(exc)}
            chunks = structured["chunks"]
            vector_chunks = structured["vector_chunks"]
            if not vector_chunks:
                raise ValueError("No readable content chunks produced")
            vector_chunks = self._validate_embedding_chunks(document.id, vector_chunks)
            doc_repo.update_status(document, "ingesting", {"failed_stage": "embed"})
            vectors = embed_texts([chunk["content"] for chunk in vector_chunks])
            doc_repo.update_status(document, "ingesting", {"failed_stage": "qdrant_upsert"})
            point_ids = QdrantVectorStore().upsert_chunks(user_id, document.id, vector_chunks, vectors, document)
        except Exception:
            raise
        doc_repo.update_status(document, "ingesting", {"failed_stage": "db_write"})
        point_by_chunk_id = {
            str(chunk.get("metadata", {}).get("chunk_id")): point_id
            for chunk, point_id in zip(vector_chunks, point_ids, strict=True)
            if chunk.get("metadata", {}).get("chunk_id")
        }
        rows = []
        for chunk in chunks:
            chunk_metadata = dict(chunk.get("metadata", {}))
            chunk_metadata.update({"char_start": chunk["char_start"], "char_end": chunk["char_end"], "heading_path": chunk.get("heading_path", [])})
            qdrant_point_id = ""
            if chunk_metadata.get("chunk_role") == "child":
                qdrant_point_id = point_by_chunk_id.get(str(chunk_metadata.get("chunk_id")), "")
            rows.append(
                {
                    "document_id": document.id,
                    "user_id": user_id,
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "token_count": chunk["token_count"],
                    "qdrant_point_id": qdrant_point_id,
                    "metadata_json": chunk_metadata,
                }
            )
        saved_chunks = chunk_repo.bulk_create_chunks(rows)
        child_chunks = [chunk for chunk in saved_chunks if (chunk.metadata_json or {}).get("chunk_role") == "child"]
        token_count = sum(chunk.token_count for chunk in child_chunks)
        child_count = len(child_chunks)
        doc_repo.update_ingest_stats(
            document,
            child_count,
            token_count,
            parsed.get("metadata", {}),
            {
                "overview": structured.get("overview", {}),
                "document_map": structured.get("document_map", {}),
                "chunking_stats": structured.get("stats", {}),
                "pg_chunk_count": len(saved_chunks),
            },
        )
        return {"status": "ingested", "chunk_count": child_count, "token_count": token_count}

    def _validate_embedding_chunks(self, document_id: int, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}
            content = chunk.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                logger.warning(
                    "document.embedding_chunk_skipped_empty document_id=%s chunk_index=%s chunk_id=%s",
                    document_id,
                    chunk.get("chunk_index"),
                    metadata.get("chunk_id"),
                )
                continue
            if not isinstance(content, str):
                raise TypeError(f"Embedding chunk content must be str: document_id={document_id} chunk_index={chunk.get('chunk_index')} chunk_id={metadata.get('chunk_id')} type={type(content).__name__}")
            content_length = len(content.strip())
            if content_length > MAX_EMBED_CHARS:
                logger.error(
                    "document.embedding_chunk_too_long document_id=%s chunk_index=%s chunk_id=%s content_length=%s max_chars=%s",
                    document_id,
                    chunk.get("chunk_index"),
                    metadata.get("chunk_id"),
                    content_length,
                    MAX_EMBED_CHARS,
                )
                raise ValueError(f"Embedding chunk too long: document_id={document_id} chunk_index={chunk.get('chunk_index')} chunk_id={metadata.get('chunk_id')} length={content_length} max={MAX_EMBED_CHARS}")
            valid_chunks.append(chunk)
        if not valid_chunks:
            raise ValueError("No non-empty chunks available for embedding")
        return valid_chunks

    def ingest_chat_document(self, db: Session, user_id: int, document_id: int) -> dict[str, Any]:
        doc_repo = DocumentRepository(db)
        chunk_repo = DocumentChunkRepository(db)
        document = doc_repo.get_by_id_for_user(user_id, document_id)
        if not document:
            raise ValueError("Document not found")
        metadata = dict(document.metadata_json or {})
        if metadata.get("kind") != "document":
            return {"document_id": document_id, "status": "skipped", "reason": "not a document type"}
        if document.status in INGESTED_STATUSES or metadata.get("ingest_status") in INGESTED_STATUSES:
            return {"document_id": document_id, "status": "skipped", "reason": "already_ingested", "chunk_count": metadata.get("chunk_count", 0)}

        try:
            ingest_result = self._ingest_document_internal(db, user_id, document)
            return {"document_id": document_id, "status": "ingested", "chunk_count": ingest_result["chunk_count"], "token_count": ingest_result["token_count"]}
        except Exception as exc:
            metadata["ingest_status"] = "failed"
            metadata["error"] = str(exc)
            metadata["failed_stage"] = (document.metadata_json or {}).get("failed_stage") or "ingest"
            doc_repo.update(document, status="failed", metadata_json=metadata)
            logger.exception("Chat document ingest failed for document_id=%s", document_id)
            return {"document_id": document_id, "status": "failed", "error": str(exc)}

    def get_chat_attachment(self, db: Session, user_id: int, document_id: int) -> Document:
        document = DocumentRepository(db).get_by_id_for_user(user_id, document_id)
        if not document:
            raise ValueError("Document not found")
        metadata = document.metadata_json or {}
        if metadata.get("upload_type") != "chat" and document.source_type not in ("chat_upload", "user_upload"):
            raise ValueError("Document is not a chat attachment")
        return document

    def _document_dir(self, user_id: int, document_id: int) -> Path:
        return Path("storage/uploads") / str(user_id) / str(document_id)


document_service = DocumentService()
