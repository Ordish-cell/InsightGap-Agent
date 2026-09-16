from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.document_service import document_service
from src.web_app.services.document_ingest_task_manager import document_ingest_task_manager
from src.web_app.services.rag_service import rag_service

router = APIRouter()


@router.post("/documents/upload")
def upload_document(file: UploadFile, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(document_service.upload_document(db, user_id, file))
    except ValueError as exc:
        return fail("DOCUMENT_UPLOAD_FAILED", str(exc))


@router.get("/documents")
def list_documents(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(document_service.list_documents(db, user_id))


@router.get("/documents/{document_id}")
def get_document(document_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(document_service.get_document(db, user_id, document_id))
    except ValueError as exc:
        return fail("DOCUMENT_NOT_FOUND", str(exc))


@router.post("/documents/{document_id}/ingest")
def ingest_document(document_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(document_service.ingest_document(db, user_id, document_id))
    except Exception as exc:
        return fail("DOCUMENT_INGEST_FAILED", str(exc))


@router.delete("/documents/{document_id}")
def delete_document(document_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(document_service.delete_document(db, user_id, document_id))
    except ValueError as exc:
        return fail("DOCUMENT_DELETE_FAILED", str(exc))


@router.post("/rag/search")
def rag_search(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(
            rag_service.search(
                user_id=user_id,
                query=payload.get("query", ""),
                top_k=payload.get("top_k", 5),
                min_score=payload.get("min_score", 0.2),
                document_ids=payload.get("document_ids") or None,
                db=db,
            )
        )
    except Exception as exc:
        return fail("RAG_SEARCH_FAILED", str(exc))


@router.post("/rag/ask")
def rag_ask(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(
            rag_service.ask(
                user_id=user_id,
                question=payload.get("question", ""),
                top_k=payload.get("top_k", 5),
                min_score=payload.get("min_score", 0.2),
                document_ids=payload.get("document_ids") or None,
                answer_mode=payload.get("answer_mode", "auto"),
                db=db,
            )
        )
    except Exception as exc:
        return fail("RAG_ASK_FAILED", str(exc))


@router.get("/rag/stats")
def rag_stats(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(rag_service.stats(db, user_id))
    except Exception as exc:
        return fail("RAG_STATS_FAILED", str(exc))


@router.post("/documents/chat-upload")
async def chat_upload(file: UploadFile, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db), model_config_id: Annotated[int | None, Form()] = None):
    try:
        if model_config_id is not None:
            from src.web_app.services.llm_registry_service import resolve_model_context
            resolve_model_context(db, user_id, model_config_id)
        result = document_service.upload_chat_attachment(db, user_id, file)
        if result.get("kind") == "document" and result.get("status") == "processing":
            if model_config_id is not None:
                from src.web_app.db.repositories.document_repository import DocumentRepository
                repo = DocumentRepository(db)
                document = repo.get_by_id_for_user(user_id, int(result["document_id"]))
                repo.update(document, metadata_json={**(document.metadata_json or {}), "summary_model_config_id": model_config_id})
            document_ingest_task_manager.start(user_id, int(result["document_id"]))
        return ok(result)
    except ValueError as exc:
        return fail("CHAT_UPLOAD_FAILED", str(exc))


@router.get("/documents/{document_id}/status")
def get_document_status(document_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(document_service.get_ingest_status(db, user_id, document_id))
    except ValueError as exc:
        return fail("DOCUMENT_NOT_FOUND", str(exc))


@router.post("/documents/{document_id}/ingest-background")
async def retry_document_ingest(document_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        result = document_service.queue_ingest_retry(db, user_id, document_id)
        document_ingest_task_manager.start(user_id, document_id)
        return ok(result)
    except ValueError as exc:
        return fail("DOCUMENT_INGEST_FAILED", str(exc))
    except Exception as exc:
        from src.web_app.core.errors import DocumentIngestError
        if isinstance(exc, DocumentIngestError):
            return fail(
                "DOCUMENT_INGEST_FAILED",
                str(exc),
                details={"document_id": exc.document_id, "detail": exc.detail, "ingest_status": "failed", "error_message": str(exc)},
            )
        return fail("CHAT_UPLOAD_FAILED", str(exc))


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        document = document_service.get_chat_attachment(db, user_id, document_id)
    except ValueError as exc:
        return fail("DOCUMENT_NOT_FOUND", str(exc))
    return FileResponse(document.file_path, media_type=document.metadata_json.get("mime_type", "application/octet-stream") if document.metadata_json else "application/octet-stream", filename=document.filename)
