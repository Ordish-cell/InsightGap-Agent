import shutil
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.web_app.core.config import settings
from src.web_app.db.repositories.document_repository import DocumentChunkRepository, DocumentRepository
from src.web_app.rag.chunker import chunk_markdown
from src.web_app.rag.document_parser import ALLOWED_EXTENSIONS, parse_document
from src.web_app.rag.embeddings import embed_texts
from src.web_app.rag.vector_store import QdrantVectorStore

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


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

    def _document_dir(self, user_id: int, document_id: int) -> Path:
        return Path("storage/uploads") / str(user_id) / str(document_id)


document_service = DocumentService()
