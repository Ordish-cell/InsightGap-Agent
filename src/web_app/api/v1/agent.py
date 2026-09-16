from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.web_app.agent.runtime.ledger_stream import stream_ledger_events
from src.web_app.agent.schemas import AgentRunRequest
from src.web_app.db.session import get_db
from src.web_app.db.session import SessionLocal
from src.web_app.db.repositories.agent_repository import AgentEventRepository, AgentRunRepository
from src.web_app.schemas.common import ok
from src.web_app.services.agent_service import (
    PendingApprovalExistsError,
    archive_conversation,
    clear_conversation,
    create_conversation as create_conversation_service,
    delete_conversation,
    get_conversation,
    get_run as get_run_data,
    hard_delete_conversation,
    list_conversations,
    list_steps,
    prepare_agent_run,
    replay_events,
    run_agent_async,
    update_conversation,
)
from src.web_app.services.agent_run_task_manager import agent_run_task_manager
from src.web_app.services.approval_service import update_approval_status
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.llm_registry_service import ModelSetupError

router = APIRouter()

from pydantic import BaseModel, Field, ConfigDict
from src.web_app.services.chat_control_service import ChatControlError, request_control


class ChatControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    client_command_id: str = Field(min_length=1, max_length=128)


class ChatSteerRequest(ChatControlRequest):
    text: str = Field(min_length=1, max_length=32000)


@router.post("/runs/{run_id}/interrupt", status_code=202)
async def interrupt_chat(run_id: int, payload: ChatControlRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(request_control(db, user_id, run_id, "interrupt", payload.client_command_id))
    except ChatControlError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.post("/runs/{run_id}/steer", status_code=202)
async def steer_chat(run_id: int, payload: ChatSteerRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(request_control(db, user_id, run_id, "steer", payload.client_command_id, payload.text))
    except ChatControlError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.post("/runs")
async def create_run(payload: AgentRunRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(await run_agent_async(db, user_id, payload.runtime_payload()))
    except ModelSetupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ChatControlError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.post("/runs/start", status_code=202)
async def start_run(payload: AgentRunRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    request_payload = payload.runtime_payload()
    request_payload["_chat_managed"] = True
    try:
        prepared = prepare_agent_run(db, user_id, request_payload)
    except ModelSetupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ChatControlError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    agent_run_task_manager.start(prepared["run_id"], user_id, payload=request_payload)
    return ok(prepared)


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


@router.delete("/conversations/{conversation_id}", deprecated=True)
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


@router.delete("/conversations/{conversation_id}/hard", status_code=202)
async def hard_delete_agent_conversation(
    conversation_id: str,
    cancel_pending: bool = False,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    try:
        import asyncio
        from src.web_app.services.conversation_deletion import request_deletion, deletion_manager
        result = await asyncio.to_thread(request_deletion, db, user_id, conversation_id, cancel_pending)
        if result["status"] == "pending":
            deletion_manager.start(result["id"])
        return ok(result)
    except PendingApprovalExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "CONVERSATION_HAS_PENDING_APPROVAL", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/deletion-tasks")
def list_deletion_tasks(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    from sqlalchemy import select
    from src.web_app.services.conversation_deletion import Job, public, require_schema
    require_schema(db)
    return ok([public(j) for j in db.scalars(select(Job).where(Job.user_id == user_id).order_by(Job.id.desc()).limit(100))])


@router.get("/deletion-tasks/{job_id}")
def get_deletion_task(job_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    from sqlalchemy import select
    from src.web_app.services.conversation_deletion import Job, public, require_schema
    require_schema(db)
    job = db.scalar(select(Job).where(Job.id == job_id, Job.user_id == user_id))
    if not job:
        raise HTTPException(404, "Deletion task not found")
    return ok(public(job))


@router.post("/deletion-tasks/{job_id}/retry", status_code=202)
async def retry_deletion_task(job_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    from sqlalchemy import select
    from src.web_app.services.conversation_deletion import Job, public, require_schema, deletion_manager
    require_schema(db)
    job = db.scalar(select(Job).where(Job.id == job_id, Job.user_id == user_id))
    if not job:
        raise HTTPException(404, "Deletion task not found")
    if job.status == "failed" and job_id not in deletion_manager.tasks:
        job.status, job.error_message = "pending", ""
        job.progress = {**(job.progress or {}), "previous_attempts": (job.progress or {}).get("previous_attempts", 0) + job.attempts}
        job.attempts = 0
        db.commit()
        deletion_manager.start(job_id)
    return ok(public(job))


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


@router.get("/runs/{run_id}/events")
def run_events(
    run_id: int,
    after_seq: int = Query(0, ge=0),
    until_seq: int | None = Query(None, ge=0),
    limit: int = Query(200, ge=1, le=500),
    event_type: str | None = None,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    try:
        return ok(replay_events(db, user_id, run_id, after_seq=after_seq, until_seq=until_seq, limit=limit, event_type=event_type))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/events/stream")
def run_events_stream(
    run_id: int,
    after_seq: int | None = Query(None, ge=0),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    if not AgentRunRepository(db).get_by_user(user_id, run_id):
        raise HTTPException(status_code=404, detail="AgentRun not found")
    try:
        header_cursor = max(0, int(last_event_id or 0))
    except ValueError:
        header_cursor = 0
    cursor = after_seq if after_seq is not None else header_cursor

    return StreamingResponse(
        stream_ledger_events(SessionLocal, user_id, run_id, after_seq=cursor),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/approvals/{approval_id}/approve")
def approve_agent_approval(approval_id: int, payload: dict | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(update_approval_status(db, user_id, approval_id, "approved", payload or {}))
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("APPROVAL_CONTEXT_GONE"):
            raise HTTPException(status_code=409, detail={"code": "APPROVAL_CONTEXT_GONE", "message": msg.split(": ", 1)[1] if ": " in msg else msg}) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/approvals/{approval_id}/reject")
def reject_agent_approval(approval_id: int, payload: dict | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(update_approval_status(db, user_id, approval_id, "rejected", payload or {}))
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("APPROVAL_CONTEXT_GONE"):
            raise HTTPException(status_code=409, detail={"code": "APPROVAL_CONTEXT_GONE", "message": msg.split(": ", 1)[1] if ": " in msg else msg}) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/runs/{run_id}/resume", status_code=202)
async def resume_run(run_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    run = AgentRunRepository(db).get_by_user(user_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="AgentRun not found")
    if run.status not in {"waiting_approval", "paused", "resuming"}:
        raise HTTPException(status_code=409, detail=f"Run is not resumable (status={run.status})")
    started = agent_run_task_manager.start(run_id, user_id, resume=True)
    return ok({"run_id": run_id, "started": started, "last_event_seq": AgentEventRepository(db).max_seq(user_id, run_id)})
