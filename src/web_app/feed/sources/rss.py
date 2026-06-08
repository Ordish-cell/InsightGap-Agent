import hashlib

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem, parse_datetime


class RSSSource(FeedSource):
    name = "rss_source"
    source_type = "blog"

    def __init__(self):
        self.enabled = settings.feed_rss_enabled and bool(settings.csv(settings.feed_rss_urls))
        self.max_items = settings.feed_rss_max_items
        self.urls = settings.csv(settings.feed_rss_urls)

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled or not self.urls:
            return []
        items: list[RawFeedItem] = []
        async with httpx.AsyncClient(timeout=8) as client:
            for url in self.urls:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    items.extend(self._parse(response.text, url))
                except Exception:
                    continue
                if len(items) >= self.max_items:
                    break
        return items[: self.max_items]

    def _parse(self, text: str, source_url: str) -> list[RawFeedItem]:
        try:
            import feedparser

            feed = feedparser.parse(text)
            source_name = feed.feed.get("title", source_url)
            return [
                RawFeedItem(
                    source_id="rss:" + hashlib.sha256((entry.get("id") or entry.get("link") or entry.get("title", "")).encode()).hexdigest(),
                    source_type="blog",
                    title=entry.get("title", ""),
                    summary=entry.get("summary", ""),
                    url=entry.get("link"),
                    published_at=parse_datetime(entry.get("published")),
                    author=entry.get("author"),
                    raw={"source_name": source_name},
                    tags=["rss", "blog", source_name],
                    domain_hints=[],
                    provider="rss",
                    source_kind="search",
                    search_query=source_url,
                )
                for entry in feed.entries
                if entry.get("title")
            ]
        except Exception:
            return []
