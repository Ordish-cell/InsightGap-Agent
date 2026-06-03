from sqlalchemy.orm import Session

from src.web_app.db.repositories.agent_repository import AgentRunRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository


def approval_to_dict(item) -> dict:
    return {"id": item.id, "user_id": item.user_id, "run_id": item.run_id, "approval_type": item.approval_type, "title": item.title, "description": item.description, "payload": item.payload or {}, "status": item.status}


def list_approvals(db: Session, user_id: int) -> list[dict]:
    return [approval_to_dict(item) for item in ApprovalRepository(db).list_by_user(user_id)]


def update_approval_status(db: Session, user_id: int, approval_id: int, status: str) -> dict:
    repo = ApprovalRepository(db)
    item = repo.get_by_user(user_id, approval_id)
    if not item:
        raise ValueError("Approval not found")
    repo.update(item, status=status)
    if status == "approved" and item.run_id:
        run = AgentRunRepository(db).get_by_user(user_id, item.run_id)
        if run:
            AgentRunRepository(db).update(run, status="completed", result_summary="Approved. Waiting for executor implementation.")
    return approval_to_dict(item)
