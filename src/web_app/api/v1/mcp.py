from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.mcp.schemas import ToolCallRequest
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.mcp_service import mcp_service

router = APIRouter()


@router.get("/tools")
def list_tools(db: Session = Depends(get_db)):
    return ok({"tools": mcp_service.list_tools(db)})


@router.get("/tools/{tool_name}")
def get_tool(tool_name: str, db: Session = Depends(get_db)):
    tool = mcp_service.get_tool(db, tool_name)
    return ok(tool) if tool else fail("TOOL_NOT_FOUND", "Tool not found")


@router.post("/tool-calls")
def call_tool(payload: ToolCallRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(mcp_service.call_tool(
        db,
        user_id,
        payload.tool_name,
        payload.input,
        agent_run_id=payload.agent_run_id,
        dry_run=payload.dry_run,
        idempotency_key=payload.idempotency_key,
        approval_mode="standalone",
    ))


@router.get("/tool-calls")
def list_tool_calls(limit: int = 50, offset: int = 0, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok({"tool_calls": mcp_service.list_tool_calls(db, user_id, limit, offset)})


@router.get("/tool-calls/{tool_call_id}")
def get_tool_call(tool_call_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return ok(mcp_service.get_tool_call(db, user_id, tool_call_id))
    except ValueError as exc:
        return fail("TOOL_CALL_NOT_FOUND", str(exc))


@router.get("/health")
def health(db: Session = Depends(get_db)):
    return ok(mcp_service.health(db))
