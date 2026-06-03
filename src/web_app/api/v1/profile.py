from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.profile_service import get_profile as get_profile_data, update_profile as update_profile_data

router = APIRouter()


@router.get("/me")
def get_profile(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(get_profile_data(db, user_id))


@router.put("/me")
def update_profile(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(update_profile_data(db, user_id, payload))
