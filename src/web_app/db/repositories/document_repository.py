from sqlalchemy import delete, select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Document, DocumentChunk


class DocumentRepository(BaseRepository[Document]):
    model = Document

    def create_uploaded_document(self, user_id: int, filename: str, file_path: str, file_type: str, metadata: dict | None = None) -> Document:
        metadata_json = dict(metadata or {})
        metadata_json.setdefault("ingest_status", "uploaded")
        return self.create(user_id=user_id, filename=filename, file_path=file_path, file_type=file_type, source_type="user_upload", status="uploaded", metadata_json=metadata_json)

    def get_by_id_for_user(self, user_id: int, document_id: int) -> Document | None:
        return self.db.execute(select(Document).where(Document.user_id == user_id, Document.id == document_id)).scalar_one_or_none()

    def list_by_user(self, user_id: int) -> list[Document]:
        return list(self.db.execute(select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())).scalars())

    def update_status(self, document: Document, status: str, metadata: dict | None = None) -> Document:
        current = dict(document.metadata_json or {})
        current["ingest_status"] = status
        if metadata:
            current.update(metadata)
        return self.update(document, status=status, metadata_json=current)

    def update_ingest_stats(self, document: Document, chunk_count: int, token_count: int, parser_metadata: dict, ingest_metadata: dict | None = None) -> Document:
        metadata = dict(document.metadata_json or {})
        metadata.update({"chunk_count": chunk_count, "token_count": token_count, "parser_metadata": parser_metadata, "ingest_status": "ingested", "error": None, "error_message": None, "failed_stage": None})
        if ingest_metadata:
            metadata.update(ingest_metadata)
        return self.update(document, status="ingested", metadata_json=metadata)

    def mark_failed(self, document: Document, error_message: str, failed_stage: str | None = None) -> Document:
        metadata = dict(document.metadata_json or {})
        metadata["error_message"] = error_message
        metadata["error"] = error_message
        metadata["ingest_status"] = "failed"
        if failed_stage:
            metadata["failed_stage"] = failed_stage
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

    def get_parent_chunk(self, user_id: int, document_id: int, parent_id: str) -> DocumentChunk | None:
        if not parent_id:
            return None
        chunks = self.list_parent_chunks(user_id, {document_id: [parent_id]})
        return chunks.get((document_id, parent_id))

    def list_parent_chunks(self, user_id: int, document_parent_ids: dict[int, list[str]]) -> dict[tuple[int, str], DocumentChunk]:
        document_ids = [int(document_id) for document_id, parent_ids in document_parent_ids.items() if parent_ids]
        if not document_ids:
            return {}
        stmt = select(DocumentChunk).where(
            DocumentChunk.user_id == user_id,
            DocumentChunk.document_id.in_(document_ids),
        )
        parents: dict[tuple[int, str], DocumentChunk] = {}
        wanted = {
            (int(document_id), str(parent_id))
            for document_id, parent_ids in document_parent_ids.items()
            for parent_id in parent_ids
            if parent_id
        }
        for chunk in self.db.execute(stmt).scalars():
            metadata = chunk.metadata_json or {}
            parent_id = str(metadata.get("chunk_id") or "")
            key = (int(chunk.document_id), parent_id)
            if metadata.get("chunk_role") == "parent" and key in wanted:
                parents[key] = chunk
        return parents

    def list_child_candidates(self, user_id: int, document_ids: list[int] | None = None, limit: int = 1000) -> list[DocumentChunk]:
        stmt = select(DocumentChunk).where(DocumentChunk.user_id == user_id).order_by(DocumentChunk.created_at.desc(), DocumentChunk.id.desc())
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_([int(item) for item in document_ids]))
        if limit and limit > 0:
            stmt = stmt.limit(limit)
        chunks: list[DocumentChunk] = []
        for chunk in self.db.execute(stmt).scalars():
            if (chunk.metadata_json or {}).get("chunk_role") == "child":
                chunks.append(chunk)
        return chunks

    def delete_by_document(self, user_id: int, document_id: int) -> int:
        result = self.db.execute(delete(DocumentChunk).where(DocumentChunk.user_id == user_id, DocumentChunk.document_id == document_id))
        self.db.commit()
        return result.rowcount or 0
