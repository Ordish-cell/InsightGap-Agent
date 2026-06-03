from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.artifact_service import artifact_service

router = APIRouter()


@router.get("")
def list_artifacts(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(artifact_service.list_artifacts(user_id, db))


@router.get("/{artifact_id}")
def get_artifact(artifact_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(artifact_service.get_artifact(artifact_id, user_id, db))
