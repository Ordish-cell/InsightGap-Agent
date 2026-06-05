from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.web_app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=3600, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> dict[str, object]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"configured": True, "available": True, "message": "Database is available"}
    except SQLAlchemyError as exc:
        return {"configured": True, "available": False, "message": str(exc)}


def check_redis_health() -> dict[str, object]:
    if not settings.redis_url:
        return {"configured": False, "available": False, "message": "Redis is not configured"}
    try:
        import redis

        client = redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return {"configured": True, "available": True, "message": "Redis is available"}
    except Exception as exc:
        return {"configured": True, "available": False, "message": str(exc)}
