from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.web_app.db.repositories.user_repository import UserRepository
from src.web_app.db.session import get_db
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id, login_user, register_user, user_to_public

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str
    password: str
    nickname: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    try:
        return ok(register_user(db, payload.email, payload.password, payload.nickname))
    except ValueError as exc:
        return fail("REGISTER_FAILED", str(exc))


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        return ok(login_user(db, payload.email, payload.password))
    except ValueError as exc:
        return fail("LOGIN_FAILED", str(exc))


@router.get("/me")
def me(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    try:
        user_id = get_current_user_id(authorization)
        user = UserRepository(db).get_by_user_id(user_id)
        return ok(user_to_public(user)) if user else fail("USER_NOT_FOUND", "User not found")
    except Exception as exc:
        return fail("INVALID_TOKEN", str(exc))
