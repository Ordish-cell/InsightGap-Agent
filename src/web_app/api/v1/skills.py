from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.skill_service import skill_service

router = APIRouter()


@router.get("")
def list_skills(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(skill_service.list_skills(user_id, db))


@router.post("/drafts")
def create_draft(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(skill_service.create_skill_draft_from_run(payload.get("run_id", 0), user_id, db, payload))


@router.post("/{skill_id}/approve")
def approve_skill(skill_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(skill_service.approve_skill(skill_id, user_id, db))


@router.post("/{skill_id}/disable")
def disable_skill(skill_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(skill_service.disable_skill(skill_id, user_id, db))
