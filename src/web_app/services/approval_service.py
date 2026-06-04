from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.db.repositories.agent_repository import AgentRunRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository


def approval_to_dict(item) -> dict:
    return {"id": item.id, "user_id": item.user_id, "run_id": item.run_id, "approval_type": item.approval_type, "title": item.title, "description": item.description, "payload": item.payload or {}, "status": item.status}


def list_approvals(db: Session, user_id: int) -> list[dict]:
    return [approval_to_dict(item) for item in ApprovalRepository(db).list_by_user(user_id)]


def update_approval_status(db: Session, user_id: int, approval_id: int, status: str, decision: dict | None = None) -> dict:
    repo = ApprovalRepository(db)
    item = repo.get_by_user(user_id, approval_id)
    if not item:
        raise ValueError("Approval not found")
    payload = dict(item.payload or {})
    if decision:
        payload["decision"] = decision
    repo.update(item, status=status, payload=payload)
    event_type = "approval_approved" if status == "approved" else "approval_rejected"
    if item.run_id:
        record_event(db, item.run_id, event_type, {"approval_id": item.id, "decision": decision or {}, "payload": payload}, user_id=user_id)
        run = AgentRunRepository(db).get_by_user(user_id, item.run_id)
        if run:
            if status == "approved":
                AgentRunRepository(db).update(run, status="completed", result_summary="Approval approved. External action remains dry-run/mock in this stage.")
                record_event(db, item.run_id, "run_completed", {"status": "completed", "approval_id": item.id, "dry_run": True}, user_id=user_id)
            else:
                AgentRunRepository(db).update(run, status="cancelled", result_summary="Approval rejected. Tool execution skipped.")
                record_event(db, item.run_id, "run_cancelled", {"status": "cancelled", "approval_id": item.id}, user_id=user_id)
    return approval_to_dict(item)
