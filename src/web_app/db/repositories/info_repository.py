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
        import logging
        _logger = logging.getLogger(__name__)
        existing = self.get_by_content_hash(values["content_hash"])
        if existing:
            metadata = dict(existing.raw_metadata or {})
            new_meta = values.get("raw_metadata") or {}
            metadata.update(new_meta)
            # Also update non-metadata fields if existing values are empty
            updates: dict = {"raw_metadata": metadata}
            for field in ("source_url", "title", "summary", "source_type"):
                existing_val = getattr(existing, field, None)
                new_val = values.get(field)
                if (not existing_val) and new_val:
                    updates[field] = new_val
            result = self.update(existing, **updates)
            _logger.info(
                "info_item metadata updated id=%s provider=%s source_kind=%s search_bucket=%s",
                existing.id,
                metadata.get("provider", ""),
                metadata.get("source_kind", ""),
                metadata.get("search_bucket", ""),
            )
            return result, False
        return self.create(**values), True

    def get_by_id(self, item_id: int) -> InfoItem | None:
        return self.db.get(InfoItem, item_id)
