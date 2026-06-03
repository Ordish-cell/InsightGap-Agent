from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.web_app.agent.runtime.events import to_sse
from src.web_app.agent.schemas import AgentRunRequest
from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.agent_service import get_run as get_run_data, list_events, list_steps, run_agent_async
from src.web_app.services.auth_service import get_current_user_id

router = APIRouter()


@router.post("/run")
async def run_agent_legacy(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(await run_agent_async(db, user_id, payload))


@router.post("/runs")
async def create_run(payload: AgentRunRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(await run_agent_async(db, user_id, payload.model_dump()))


@router.get("/runs/{run_id}")
def get_run(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(get_run_data(db, user_id, run_id))


@router.get("/runs/{run_id}/steps")
def get_steps(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok({"run_id": run_id, "steps": list_steps(db, user_id, run_id)})


@router.get("/runs/{run_id}/stream")
def stream_run(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    def events():
        yield from to_sse([{"event": "status", "data": {"run_id": run_id, "status": get_run_data(db, user_id, run_id)["status"]}}])
        yield from to_sse(list_events(db, user_id, run_id))

    return StreamingResponse(events(), media_type="text/event-stream")
