from sqlalchemy import delete, select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Document, DocumentChunk


class DocumentRepository(BaseRepository[Document]):
    model = Document

    def create_uploaded_document(self, user_id: int, filename: str, file_path: str, file_type: str, metadata: dict | None = None) -> Document:
        return self.create(user_id=user_id, filename=filename, file_path=file_path, file_type=file_type, source_type="user_upload", status="uploaded", metadata_json=metadata or {})

    def get_by_id_for_user(self, user_id: int, document_id: int) -> Document | None:
        return self.db.execute(select(Document).where(Document.user_id == user_id, Document.id == document_id)).scalar_one_or_none()

    def list_by_user(self, user_id: int) -> list[Document]:
        return list(self.db.execute(select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())).scalars())

    def update_status(self, document: Document, status: str, metadata: dict | None = None) -> Document:
        current = dict(document.metadata_json or {})
        if metadata:
            current.update(metadata)
        return self.update(document, status=status, metadata_json=current)

    def update_ingest_stats(self, document: Document, chunk_count: int, token_count: int, parser_metadata: dict) -> Document:
        metadata = dict(document.metadata_json or {})
        metadata.update({"chunk_count": chunk_count, "token_count": token_count, "parser_metadata": parser_metadata})
        return self.update(document, status="ingested", metadata_json=metadata)

    def mark_failed(self, document: Document, error_message: str) -> Document:
        metadata = dict(document.metadata_json or {})
        metadata["error_message"] = error_message
        return self.update(document, status="failed", metadata_json=metadata)


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    model = DocumentChunk

    def bulk_create_chunks(self, rows: list[dict]) -> list[DocumentChunk]:
        chunks = [DocumentChunk(**row) for row in rows]
        self.db.add_all(chunks)
        self.db.commit()
        for chunk in chunks:
            self.db.refresh(chunk)
        return chunks

    def list_by_document(self, user_id: int, document_id: int) -> list[DocumentChunk]:
        stmt = select(DocumentChunk).where(DocumentChunk.user_id == user_id, DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index)
        return list(self.db.execute(stmt).scalars())

    def get_chunk_by_id(self, user_id: int, chunk_id: int) -> DocumentChunk | None:
        return self.db.execute(select(DocumentChunk).where(DocumentChunk.user_id == user_id, DocumentChunk.id == chunk_id)).scalar_one_or_none()

    def delete_by_document(self, user_id: int, document_id: int) -> int:
        result = self.db.execute(delete(DocumentChunk).where(DocumentChunk.user_id == user_id, DocumentChunk.document_id == document_id))
        self.db.commit()
        return result.rowcount or 0
