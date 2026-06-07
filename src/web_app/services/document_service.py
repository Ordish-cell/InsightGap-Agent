import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.web_app.core.config import settings
from src.web_app.db.repositories.document_repository import DocumentChunkRepository, DocumentRepository
from src.web_app.models.orm import Document
from src.web_app.rag.chunker import chunk_markdown
from src.web_app.rag.document_parser import ALLOWED_EXTENSIONS, parse_document
from src.web_app.rag.embeddings import embed_texts
from src.web_app.rag.vector_store import QdrantVectorStore

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CHAT_UPLOAD_BYTES = getattr(settings, "max_chat_upload_bytes", None) or 20 * 1024 * 1024

ALLOWED_CHAT_UPLOAD_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".json", ".html", ".htm",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

logger = logging.getLogger(__name__)


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
                    repo.mark_failed(document, "File is too large")
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
            doc_repo.update_status(document, "ingesting")
            parsed = parse_document(document.file_path, document.filename, document.file_type)
            chunks = chunk_markdown(parsed["markdown"] or parsed["text"])
            if not chunks:
                raise ValueError("No readable content chunks produced")
            vectors = embed_texts([chunk["content"] for chunk in chunks])
            point_ids = QdrantVectorStore().upsert_chunks(user_id, document.id, chunks, vectors, document)
            chunk_repo.delete_by_document(user_id, document.id)
            rows = []
            for chunk, point_id in zip(chunks, point_ids, strict=True):
                metadata = dict(chunk.get("metadata", {}))
                metadata.update({"char_start": chunk["char_start"], "char_end": chunk["char_end"], "heading_path": chunk.get("heading_path", [])})
                rows.append(
                    {
                        "document_id": document.id,
                        "user_id": user_id,
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "token_count": chunk["token_count"],
                        "qdrant_point_id": point_id,
                        "metadata_json": metadata,
                    }
                )
            saved_chunks = chunk_repo.bulk_create_chunks(rows)
            token_count = sum(chunk.token_count for chunk in saved_chunks)
            doc_repo.update_ingest_stats(document, len(saved_chunks), token_count, parsed["metadata"])
            return {"document": document_to_dict(document), "chunk_count": len(saved_chunks), "token_count": token_count}
        except Exception as exc:
            doc_repo.mark_failed(document, str(exc))
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
        QdrantVectorStore().delete_document(user_id, document_id)
        DocumentChunkRepository(db).delete_by_document(user_id, document_id)
        path = Path(document.file_path)
        if path.exists():
            shutil.rmtree(path.parent, ignore_errors=True)
        db.delete(document)
        db.commit()
        return {"deleted": True, "document_id": document_id}

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
                "ingest_status": "pending" if kind == "document" else "completed",
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
                    repo.mark_failed(document, "File is too large")
                    target.unlink(missing_ok=True)
                    raise ValueError("File is too large")
                handle.write(chunk)

        metadata = dict(document.metadata_json or {})
        metadata["size"] = size
        repo.update(document, file_path=str(target), metadata_json=metadata)
        return {
            "document_id": document.id,
            "filename": filename,
            "file_type": suffix.lstrip("."),
            "mime_type": file.content_type or "",
            "kind": kind,
            "size": size,
            "preview_url": f"/api/v1/documents/{document.id}/file",
            "status": "uploaded",
            "ingest_status": metadata.get("ingest_status", "pending"),
        }

    def ingest_chat_document(self, db: Session, user_id: int, document_id: int) -> dict[str, Any]:
        doc_repo = DocumentRepository(db)
        chunk_repo = DocumentChunkRepository(db)
        document = doc_repo.get_by_id_for_user(user_id, document_id)
        if not document:
            raise ValueError("Document not found")
        metadata = dict(document.metadata_json or {})
        if metadata.get("kind") != "document":
            return {"document_id": document_id, "status": "skipped", "reason": "not a document type"}
        if metadata.get("ingest_status") == "completed":
            return {"document_id": document_id, "status": "skipped", "reason": "already_ingested", "chunk_count": metadata.get("chunk_count", 0)}

        try:
            metadata["ingest_status"] = "processing"
            doc_repo.update(document, metadata_json=metadata)

            parsed = parse_document(document.file_path, document.filename, document.file_type)
            chunks = chunk_markdown(parsed["markdown"] or parsed["text"])
            if not chunks:
                raise ValueError("No readable content chunks produced")
            vectors = embed_texts([chunk["content"] for chunk in chunks])
            point_ids = QdrantVectorStore().upsert_chunks(user_id, document.id, chunks, vectors, document)
            chunk_repo.delete_by_document(user_id, document.id)
            rows = []
            for chunk, point_id in zip(chunks, point_ids, strict=True):
                chunk_metadata = dict(chunk.get("metadata", {}))
                chunk_metadata.update({"char_start": chunk["char_start"], "char_end": chunk["char_end"], "heading_path": chunk.get("heading_path", [])})
                rows.append(
                    {
                        "document_id": document.id,
                        "user_id": user_id,
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "token_count": chunk["token_count"],
                        "qdrant_point_id": point_id,
                        "metadata_json": chunk_metadata,
                    }
                )
            saved_chunks = chunk_repo.bulk_create_chunks(rows)
            token_count = sum(chunk.token_count for chunk in saved_chunks)
            metadata["ingest_status"] = "completed"
            metadata["chunk_count"] = len(saved_chunks)
            metadata["token_count"] = token_count
            metadata["parser_metadata"] = parsed.get("metadata", {})
            doc_repo.update(document, status="ingested", metadata_json=metadata)
            return {"document_id": document_id, "status": "ingested", "chunk_count": len(saved_chunks), "token_count": token_count}
        except Exception as exc:
            metadata["ingest_status"] = "failed"
            metadata["error"] = str(exc)
            doc_repo.update(document, metadata_json=metadata)
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
