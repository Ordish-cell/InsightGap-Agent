"""Approval service — manages approval lifecycle.

After user approves/rejects, this service:
1. Validates the approval belongs to the user and is pending.
2. Marks the approval as approved or rejected.
3. Records audit events.
4. Returns resume info (run_id, resume_stream_url) so the frontend
   can connect to the resume SSE endpoint.

The tool is NOT executed here — execution happens inside the resume
stream (agent_service.resume_run_after_approval).
"""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.db.repositories.agent_repository import AgentRunRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository

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
        "idempotency_key": item.idempotency_key,
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
    """Approve or reject an approval.  Returns the updated approval dict with
    resume info so the frontend can connect to the resume SSE endpoint.

    Does NOT execute the tool — that happens in the resume stream.

    Raises ValueError with code APPROVAL_CONTEXT_GONE if the run, conversation,
    or assistant message no longer exists.
    """
    repo = ApprovalRepository(db)
    item = repo.get_by_user(user_id, approval_id)
    if not item:
        raise ValueError("Approval not found")
    if item.status != "pending":
        raise ValueError(f"Approval is already {item.status}")

    payload = dict(item.payload or {})
    if not item.run_id:
        now = datetime.now(UTC).replace(tzinfo=None)
        payload["decided_at"] = now.isoformat()
        payload["decided_by"] = user_id
        payload["executed"] = False
        tool_result: dict[str, Any] | None = None
        tool_call_id = payload.get("tool_call_id")
        tool_name = str(payload.get("tool_name") or "")
        if status == "approved" and tool_call_id and tool_name:
            from src.web_app.db.repositories.mcp_repository import ToolCallRepository
            from src.web_app.mcp.tool_executor import tool_executor

            tool_call = ToolCallRepository(db).get_by_user(user_id, int(tool_call_id))
            if not tool_call:
                raise ValueError("APPROVAL_CONTEXT_GONE: ToolCall not found.")
            tool_result = tool_executor.execute_approved_tool_once(
                db,
                user_id,
                int(tool_call_id),
                tool_name,
                tool_call.input or payload.get("tool_args", {}),
                agent_run_id=None,
            )
            payload["executed"] = tool_result.get("success") is True
        elif status == "rejected" and tool_call_id:
            from src.web_app.db.repositories.mcp_repository import ToolCallRepository

            ToolCallRepository(db).update_status(int(tool_call_id), "rejected", error_message="User rejected the approval")

        repo.update(item, status=status, payload=payload)
        result = approval_to_dict(item)
        if tool_result is not None:
            result["tool_result"] = tool_result
        return result

    # ── Validate context still exists ──────────────────────────
    from src.web_app.db.repositories.agent_repository import (
        AgentChatMessageRepository,
        AgentConversationRepository,
        AgentRunRepository,
    )
    run_repo = AgentRunRepository(db)
    conv_repo = AgentConversationRepository(db)
    msg_repo = AgentChatMessageRepository(db)

    run_id = item.run_id
    if not run_id:
        raise ValueError("APPROVAL_CONTEXT_GONE: 这个审批所属的运行已经不存在。")

    run = run_repo.get_by_user(user_id, run_id)
    if not run:
        raise ValueError("APPROVAL_CONTEXT_GONE: 这个审批所属的运行已经不存在。")
    if run.status not in ("waiting_approval", "paused"):
        raise ValueError(f"APPROVAL_CONTEXT_GONE: 运行状态已变更为 {run.status}。")

    conversation_id = run.conversation_id
    if not conversation_id:
        raise ValueError("APPROVAL_CONTEXT_GONE: 这个审批所属的会话已经不存在。")

    conversation = conv_repo.get_by_conversation_id(user_id, conversation_id)
    if not conversation or conversation.status == "deleted":
        raise ValueError("APPROVAL_CONTEXT_GONE: 这个审批所属的会话已被删除。")

    # Verify assistant message exists and is waiting
    messages = msg_repo.list_by_conversation(user_id, conversation_id)
    assistant_msg = next(
        (m for m in messages if m.run_id == run_id and m.role == "assistant"),
        None,
    )
    if not assistant_msg or assistant_msg.status != "waiting_approval":
        raise ValueError("APPROVAL_CONTEXT_GONE: 助手消息状态不匹配。")

    payload = dict(item.payload or {})
    tool_name = payload.get("tool_name", "")
    tool_call_id = payload.get("tool_call_id")

    now = datetime.now(UTC).replace(tzinfo=None)
    payload["decided_at"] = now.isoformat()
    payload["decided_by"] = user_id
    payload["executed"] = False  # will be executed during resume streaming

    event_type = "approval_approved" if status == "approved" else "approval_rejected"

    repo.update(item, status=status, payload=payload)

    record_event(
        db,
        run_id,
        event_type,
        {
            "approval_id": item.id,
            "decision": decision or {},
            "status": status,
            "tool_name": tool_name,
        },
        user_id=user_id,
    )

    if status == "rejected":
        run_repo.update(run, status="waiting_approval")  # resume stream handles final

    result = approval_to_dict(item)
    result["run_id"] = run_id
    result["resume_url"] = f"/api/v1/agent/runs/{run_id}/resume"
    return result
