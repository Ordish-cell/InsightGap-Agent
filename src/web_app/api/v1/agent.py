from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.web_app.agent.runtime.events import to_sse
from src.web_app.agent.schemas import AgentRunRequest
from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.agent_service import (
    archive_conversation,
    clear_conversation,
    create_conversation as create_conversation_service,
    delete_conversation,
    get_conversation,
    get_run as get_run_data,
    list_conversations,
    list_events,
    list_steps,
    run_agent_async,
    update_conversation,
)
from src.web_app.services.approval_service import update_approval_status
from src.web_app.services.auth_service import get_current_user_id

router = APIRouter()


@router.post("/run")
async def run_agent_legacy(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(await run_agent_async(db, user_id, payload))


@router.post("/runs")
async def create_run(payload: AgentRunRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(await run_agent_async(db, user_id, payload.model_dump()))


@router.post("/conversations")
def create_conversation(payload: dict | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(create_conversation_service(db, user_id, payload or {}))


@router.get("/conversations")
def conversations(status: str = "active", limit: int = 50, offset: int = 0, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(list_conversations(db, user_id, status=status, limit=limit, offset=offset))


@router.get("/conversations/{conversation_id}")
def conversation_detail(conversation_id: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(get_conversation(db, user_id, conversation_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: dict | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(update_conversation(db, user_id, conversation_id, payload or {}))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/archive")
def archive_agent_conversation(conversation_id: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(archive_conversation(db, user_id, conversation_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/conversations/{conversation_id}")
def delete_agent_conversation(conversation_id: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(delete_conversation(db, user_id, conversation_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/clear")
def clear_agent_conversation(conversation_id: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(clear_conversation(db, user_id, conversation_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
def get_run(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(get_run_data(db, user_id, run_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/steps")
def get_steps(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok({"run_id": run_id, "steps": list_steps(db, user_id, run_id)})
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/stream")
def stream_run(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    def events():
        yield from to_sse([{"event": "status", "data": {"run_id": run_id, "status": get_run_data(db, user_id, run_id)["status"]}}])
        yield from to_sse(list_events(db, user_id, run_id))

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/runs/{run_id}/events")
def run_events(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    def events():
        yield from to_sse(list_events(db, user_id, run_id))

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/approvals/{approval_id}/approve")
def approve_agent_approval(approval_id: int, payload: dict | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(update_approval_status(db, user_id, approval_id, "approved", payload or {}))


@router.post("/approvals/{approval_id}/reject")
def reject_agent_approval(approval_id: int, payload: dict | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(update_approval_status(db, user_id, approval_id, "rejected", payload or {}))
