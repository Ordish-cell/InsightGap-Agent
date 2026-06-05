from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.feed_service import feedback as feedback_data
from src.web_app.services.feed_service import get_card_detail, list_cards as list_feed_cards, list_home_cards, refresh_feed as refresh_feed_data, source_health, stats as feed_stats
from src.web_app.research.schemas import ResearchRequest
from src.web_app.services.research_service import research_service

router = APIRouter()


@router.post("/refresh")
def refresh_feed(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(refresh_feed_data(db, user_id))
    except Exception as exc:
        return fail("FEED_REFRESH_FAILED", str(exc))


@router.get("/cards")
def list_cards(status: str | None = None, exposure_bucket: str | None = None, relation_type: str | None = None, source_type: str | None = None, domain: str | None = None, limit: int = 20, offset: int = 0, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(list_feed_cards(db, user_id, status, exposure_bucket or relation_type, limit, offset, source_type, domain))


@router.get("/home")
def home_cards(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(list_home_cards(db, user_id))


@router.get("/cards/{card_id}")
def card_detail(card_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(get_card_detail(db, user_id, card_id))
    except ValueError as exc:
        return fail("FEED_CARD_NOT_FOUND", str(exc))


@router.post("/cards/{card_id}/feedback")
def card_feedback(card_id: int, payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        action = payload.get("action") or payload.get("status", "open")
        return ok(feedback_data(db, user_id, card_id, action, payload.get("metadata", {})))
    except ValueError as exc:
        return fail("FEED_FEEDBACK_FAILED", str(exc))


@router.post("/cards/{card_id}/research")
async def research_card(card_id: int, payload: dict | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(await research_service.research_feed_card(db, user_id, card_id, ResearchRequest(**(payload or {}))))
    except ValueError as exc:
        return fail("FEED_RESEARCH_FAILED", str(exc))


@router.get("/sources")
def feed_sources():
    return ok(source_health())


@router.get("/stats")
def stats(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(feed_stats(db, user_id))
