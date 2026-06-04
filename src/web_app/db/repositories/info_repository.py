from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import InfoItem, InfoSource


class InfoSourceRepository(BaseRepository[InfoSource]):
    model = InfoSource


class InfoItemRepository(BaseRepository[InfoItem]):
    model = InfoItem

    def get_by_content_hash(self, content_hash: str) -> InfoItem | None:
        return self.db.execute(select(InfoItem).where(InfoItem.content_hash == content_hash).order_by(InfoItem.id).limit(1)).scalar_one_or_none()

    def upsert_by_hash(self, **values) -> tuple[InfoItem, bool]:
        existing = self.get_by_content_hash(values["content_hash"])
        if existing:
            metadata = dict(existing.raw_metadata or {})
            metadata.update(values.get("raw_metadata") or {})
            return self.update(existing, raw_metadata=metadata), False
        return self.create(**values), True

    def get_by_id(self, item_id: int) -> InfoItem | None:
        return self.db.get(InfoItem, item_id)
