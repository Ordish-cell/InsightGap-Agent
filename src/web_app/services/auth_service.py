from fastapi import Header, HTTPException
from sqlalchemy.orm import Session

from src.web_app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from src.web_app.db.repositories.profile_repository import ProfileRepository
from src.web_app.db.repositories.user_repository import UserRepository
from src.web_app.db.session import SessionLocal
from src.web_app.models.orm import User


def user_to_public(user: User) -> dict:
    return {"id": user.id, "email": user.email, "nickname": user.nickname, "status": user.status}


def register_user(db: Session, email: str, password: str, nickname: str = "") -> dict:
    users = UserRepository(db)
    if users.get_by_email(email):
        raise ValueError("Email already registered")
    user = users.create(email=email, hashed_password=hash_password(password), nickname=nickname)
    ProfileRepository(db).get_or_create_default(user.id)
    return user_to_public(user)


def login_user(db: Session, email: str, password: str) -> dict:
    user = UserRepository(db).get_by_email(email)
    if not user or not verify_password(password, user.hashed_password):
        raise ValueError("Invalid email or password")
    return {"access_token": create_access_token(str(user.id)), "token_type": "bearer"}


def get_current_user_id(authorization: str | None = Header(default=None)) -> int:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    token = authorization.replace("Bearer ", "")
    try:
        return int(decode_access_token(token)["sub"])
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def get_current_user(authorization: str | None = Header(default=None)) -> User:
    user_id = get_current_user_id(authorization)
    with SessionLocal() as db:
        user = UserRepository(db).get_by_user_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
