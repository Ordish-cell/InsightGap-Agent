import hashlib

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem


class DuckDuckGoSource(FeedSource):
    name = "duckduckgo_source"
    source_type = "web"

    def __init__(self):
        self.enabled = settings.feed_duckduckgo_enabled
        self.max_items = settings.feed_duckduckgo_max_items

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        try:
            from duckduckgo_search import DDGS

            rows = DDGS().text("latest LangGraph Agent RAG MCP", region=settings.feed_duckduckgo_region, safesearch=settings.feed_duckduckgo_safesearch, timelimit=settings.feed_duckduckgo_time, max_results=self.max_items)
            return [
                RawFeedItem(source_id="duckduckgo:" + hashlib.sha256(row.get("href", "").encode()).hexdigest(), source_type="web", title=row.get("title", ""), summary=row.get("body", ""), url=row.get("href"), raw=row, tags=["duckduckgo", "web_search"], domain_hints=[])
                for row in rows
                if row.get("title")
            ]
        except Exception:
            return []
