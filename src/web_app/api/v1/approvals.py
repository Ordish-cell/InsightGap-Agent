from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.approval_service import list_approvals as list_approval_data, update_approval_status
from src.web_app.services.auth_service import get_current_user_id

router = APIRouter()


@router.get("")
def list_approvals(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(list_approval_data(db, user_id))


@router.post("/{approval_id}/approve")
def approve(approval_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(update_approval_status(db, user_id, approval_id, "approved"))


@router.post("/{approval_id}/reject")
def reject(approval_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(update_approval_status(db, user_id, approval_id, "rejected"))
