from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.document_service import document_service
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
def rag_search(payload: dict, user_id: int = Depends(get_current_user_id)):
    try:
        return ok(
            rag_service.search(
                user_id=user_id,
                query=payload.get("query", ""),
                top_k=payload.get("top_k", 5),
                min_score=payload.get("min_score", 0.2),
                document_ids=payload.get("document_ids") or None,
            )
        )
    except Exception as exc:
        return fail("RAG_SEARCH_FAILED", str(exc))


@router.post("/rag/ask")
def rag_ask(payload: dict, user_id: int = Depends(get_current_user_id)):
    try:
        return ok(
            rag_service.ask(
                user_id=user_id,
                question=payload.get("question", ""),
                top_k=payload.get("top_k", 5),
                min_score=payload.get("min_score", 0.2),
                document_ids=payload.get("document_ids") or None,
                answer_mode=payload.get("answer_mode", "auto"),
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
