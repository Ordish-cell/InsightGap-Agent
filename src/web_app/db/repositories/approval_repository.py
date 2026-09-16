from sqlalchemy import select, update

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Approval


class ApprovalRepository(BaseRepository[Approval]):
    model = Approval

    def decide_pending(self, item, status, payload):
        """Approve/reject at most once, including requests from other workers."""
        changed = self.db.execute(update(Approval).where(Approval.id == item.id,
            Approval.user_id == item.user_id, Approval.status == "pending").values(status=status, payload=payload))
        self.db.commit()
        self.db.refresh(item)
        if changed.rowcount != 1:
            raise ValueError(f"Approval is already {item.status}")
        return item

    def list_by_user(self, user_id: int) -> list[Approval]:
        return list(self.db.execute(select(Approval).where(Approval.user_id == user_id).order_by(Approval.id.desc())).scalars())

    def get_by_user(self, user_id: int, approval_id: int) -> Approval | None:
        return self.db.execute(select(Approval).where(Approval.user_id == user_id, Approval.id == approval_id)).scalar_one_or_none()

    def list_by_run(self, run_id: int) -> list[Approval]:
        return list(self.db.execute(select(Approval).where(Approval.run_id == run_id)).scalars())

    def get_by_idempotency_key(self, idempotency_key: str) -> Approval | None:
        return self.db.execute(select(Approval).where(Approval.idempotency_key == idempotency_key)).scalar_one_or_none()
