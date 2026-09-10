from typing import Any, Generic, TypeVar

from sqlalchemy.orm import Session
from sqlalchemy import select
from src.web_app.services.deletion_guard import guarded_transition

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, db: Session):
        self.db = db

    def _commit(self) -> None:
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def create(self, **values: Any) -> ModelT:
        obj = self.model(**values)
        self.db.add(obj)
        self._commit()
        self.db.refresh(obj)
        return obj

    def get_by_id(self, obj_id: int) -> ModelT | None:
        return self.db.get(self.model, obj_id)

    @guarded_transition
    def update(self, obj: ModelT, **values: Any) -> ModelT:
        # Read persisted reservations as callers may hold a stale identity-map object.
        columns = [getattr(self.model, key) for key in ("status", "metadata_json") if hasattr(self.model, key)]
        with self.db.no_autoflush:
            current = self.db.execute(select(*columns).where(self.model.id == obj.id)).mappings().first() if columns else None
        reserved = current and (current.get("status") == "deleting" or (current.get("metadata_json") or {}).get("deletion_task_id"))
        if reserved or getattr(obj, "status", None) == "deleting" or (getattr(obj, "metadata_json", None) or {}).get("deletion_task_id"):
            from src.web_app.services.deletion_guard import ConversationDeletingError
            raise ConversationDeletingError("资源正在删除，不能继续修改。")
        for key, value in values.items():
            if value is not None and hasattr(obj, key):
                setattr(obj, key, value)
        self._commit()
        self.db.refresh(obj)
        return obj
