from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Approval


class ApprovalRepository(BaseRepository[Approval]):
    model = Approval

    def list_by_user(self, user_id: int) -> list[Approval]:
        return list(self.db.execute(select(Approval).where(Approval.user_id == user_id).order_by(Approval.id.desc())).scalars())

    def get_by_user(self, user_id: int, approval_id: int) -> Approval | None:
        return self.db.execute(select(Approval).where(Approval.user_id == user_id, Approval.id == approval_id)).scalar_one_or_none()

    def list_by_run(self, run_id: int) -> list[Approval]:
        return list(self.db.execute(select(Approval).where(Approval.run_id == run_id)).scalars())

    def get_by_idempotency_key(self, idempotency_key: str) -> Approval | None:
        return self.db.execute(select(Approval).where(Approval.idempotency_key == idempotency_key)).scalar_one_or_none()
