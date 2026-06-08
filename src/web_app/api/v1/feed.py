from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

import logging

from src.web_app.db.session import get_db
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.db.repositories.feed_repository import FeedRepository
from src.web_app.services.feed_service import feedback as feedback_data
from src.web_app.services.feed_service import get_card_detail, list_cards as list_feed_cards, list_home_cards, maybe_refresh_for_user, source_health, stats as feed_stats
from src.web_app.research.schemas import ResearchRequest
from src.web_app.services.research_service import research_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/refresh")
def refresh_feed(force: bool = Query(default=True), user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    logger.warning("manual feed refresh requested user=%s", user_id)
    try:
        result = maybe_refresh_for_user(db, user_id, force=force)
        return ok(result)
    except Exception as exc:
        return fail("FEED_REFRESH_FAILED", str(exc))


@router.get("/cards")
def list_cards(status: str | None = None, exposure_bucket: str | None = None, relation_type: str | None = None, source_type: str | None = None, domain: str | None = None, limit: int = 20, offset: int = 0, all: bool = False, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(list_feed_cards(db, user_id, status, exposure_bucket or relation_type, limit, offset, source_type, domain, all=all))


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
    """Create a research run from a feed card and return the run_id immediately.

    The research executes in the background.  The frontend should navigate
    to ``/research/{run_id}`` to poll for results.
    """
    try:
        request = ResearchRequest(**(payload or {}), source="feed_card", feed_card_id=card_id, auto_start=True)
        feed_card = FeedRepository(db).get_by_user(user_id, card_id)
        if not feed_card:
            return fail("FEED_CARD_NOT_FOUND", "Feed card not found")
        result = research_service.create_research_run(db, user_id, request, feed_card=feed_card)
        return ok(result)
    except ValueError as exc:
        return fail("FEED_RESEARCH_FAILED", str(exc))


@router.get("/sources")
def feed_sources():
    return ok(source_health())


@router.get("/stats")
def stats(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(feed_stats(db, user_id))
