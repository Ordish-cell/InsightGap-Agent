from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.memory_service import memory_service

router = APIRouter()


@router.post("/add")
def add_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.add_memory(user_id, payload.get("content", ""), payload.get("memory_type", "working"), payload.get("importance", 0.0), payload.get("metadata", {}), db))


@router.post("/search")
def search_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.search_memory(user_id, payload.get("query", ""), payload.get("memory_type"), payload.get("min_importance", 0.0), db))


@router.post("/consolidate")
def consolidate_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.consolidate_memory(user_id, db))


@router.post("/forget")
def forget_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.forget_memory(user_id, payload.get("memory_id"), db))


@router.get("/summary")
def memory_summary(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.summarize_memory(user_id, db))
