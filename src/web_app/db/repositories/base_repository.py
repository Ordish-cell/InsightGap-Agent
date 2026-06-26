from typing import Any, Generic, TypeVar

from sqlalchemy.orm import Session

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

    def update(self, obj: ModelT, **values: Any) -> ModelT:
        for key, value in values.items():
            if value is not None and hasattr(obj, key):
                setattr(obj, key, value)
        self._commit()
        self.db.refresh(obj)
        return obj
