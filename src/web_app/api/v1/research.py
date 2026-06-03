from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.research.schemas import ResearchRequest
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.research_service import research_service

router = APIRouter()


@router.post("/runs")
async def create_research_run(payload: ResearchRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(await research_service.research_query(db, user_id, payload))
    except ValueError as exc:
        return fail("RESEARCH_CREATE_FAILED", str(exc))


@router.get("/runs")
def list_research_runs(limit: int = 20, offset: int = 0, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(research_service.list_research_runs(db, user_id, limit, offset))


@router.get("/runs/{research_run_id}")
def get_research_run(research_run_id: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(research_service.get_research_run(db, user_id, research_run_id))
    except ValueError as exc:
        return fail("RESEARCH_NOT_FOUND", str(exc))


@router.post("/deep")
async def deep_research(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    request = ResearchRequest(**payload)
    try:
        return ok(await research_service.research_query(db, user_id, request))
    except ValueError as exc:
        return fail("RESEARCH_CREATE_FAILED", str(exc))


@router.get("/{run_id}")
def legacy_get_research(run_id: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(research_service.get_research_run(db, user_id, run_id))
    except ValueError as exc:
        return fail("RESEARCH_NOT_FOUND", str(exc))
