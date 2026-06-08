"""Approval service — manages approval lifecycle and executes tools after approval."""

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository, AgentRunRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.mcp.audit import tool_call_to_read
from src.web_app.mcp.tool_executor import tool_executor

logger = logging.getLogger(__name__)


def approval_to_dict(item) -> dict:
    payload = item.payload or {}
    return {
        "id": item.id,
        "user_id": item.user_id,
        "run_id": item.run_id,
        "approval_type": item.approval_type,
        "title": item.title,
        "description": item.description,
        "payload": payload,
        "status": item.status,
        "risk_level": payload.get("risk_level", ""),
        "tool_name": payload.get("tool_name", ""),
        "tool_args": payload.get("tool_args", {}),
        "preview": payload.get("preview", {}),
        "safety_notes": payload.get("safety_notes", []),
        "created_at": item.created_at.isoformat() if item.created_at else "",
        "updated_at": item.updated_at.isoformat() if item.updated_at else "",
    }


def list_approvals(db: Session, user_id: int) -> list[dict]:
    return [approval_to_dict(item) for item in ApprovalRepository(db).list_by_user(user_id)]


def update_approval_status(
    db: Session,
    user_id: int,
    approval_id: int,
    status: str,
    decision: dict | None = None,
) -> dict:
    """Approve or reject an approval. On approve, execute the tool if one is pending."""
    repo = ApprovalRepository(db)
    item = repo.get_by_user(user_id, approval_id)
    if not item:
        raise ValueError("Approval not found")
    if item.status != "pending":
        raise ValueError(f"Approval is already {item.status}")

    payload = dict(item.payload or {})
    tool_call_id = payload.get("tool_call_id")
    tool_name = payload.get("tool_name", "")
    tool_args = payload.get("tool_args") or payload.get("input") or {}

    now = datetime.now(UTC).replace(tzinfo=None)
    payload["decided_at"] = now.isoformat()
    payload["decided_by"] = user_id

    event_type = "approval_approved" if status == "approved" else "approval_rejected"

    if status == "approved" and tool_call_id and tool_name:
        # Execute the tool now that approval is granted
        try:
            result = tool_executor.execute_approved_tool(
                db, user_id, tool_call_id, tool_name, tool_args, agent_run_id=item.run_id
            )
            payload["result"] = result
            payload["executed"] = True
            payload["executed_at"] = now.isoformat()
        except Exception as exc:
            logger.exception("Tool execution after approval failed: %s", exc)
            payload["result"] = {"status": "failed", "error": str(exc)}
            payload["executed"] = False
            payload["executed_at"] = now.isoformat()

    repo.update(item, status=status, payload=payload)

    if item.run_id:
        record_event(
            db, item.run_id, event_type,
            {"approval_id": item.id, "decision": decision or {}, "payload": payload},
            user_id=user_id,
        )
        run_repo = AgentRunRepository(db)
        run = run_repo.get_by_user(user_id, item.run_id)
        if run:
            if status == "approved":
                executed = payload.get("executed", False)
                result_msg = payload.get("result", {}).get("message", "")
                summary = f"已批准并执行: {tool_name}. {result_msg}" if executed else f"已批准但执行失败: {tool_name}"
                run_repo.update(run, status="completed", result_summary=summary)
                # Append assistant message for the conversation
                _append_assistant_message(db, user_id, run, summary, payload)
            else:
                run_repo.update(run, status="cancelled", result_summary=f"已取消: {tool_name}")
                _append_assistant_message(
                    db, user_id, run,
                    f"已取消该操作，没有执行 {tool_name}。",
                    payload,
                )

    return approval_to_dict(item)


def _append_assistant_message(
    db: Session,
    user_id: int,
    run,
    content: str,
    payload: dict,
) -> None:
    """Write an assistant message to the conversation so the UI shows the decision."""
    try:
        from uuid import uuid4

        repo = AgentChatMessageRepository(db)
        repo.create(
            message_id=str(uuid4()),
            conversation_id=run.conversation_id,
            user_id=user_id,
            run_id=run.id,
            thread_id=run.thread_id,
            role="assistant",
            content=content,
            status="completed",
            metadata_json={"approval_result": True, "payload": payload},
        )
    except Exception as exc:
        logger.exception("Failed to append assistant message after approval: %s", exc)
