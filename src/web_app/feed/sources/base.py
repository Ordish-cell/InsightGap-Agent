from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RawFeedItem:
    source_id: str
    source_type: str
    title: str
    summary: str = ""
    url: str | None = None
    published_at: datetime | None = None
    author: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    domain_hints: list[str] = field(default_factory=list)
    search_bucket: str = ""  # explicit_related, adjacent_domain, or far_domain


class FeedSource:
    name = "base"
    source_type = "unknown"
    enabled = True
    max_items = 10

    async def fetch(self) -> list[RawFeedItem]:
        return []

    def health(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "status": "ok" if self.enabled else "disabled", "source_type": self.source_type}


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
