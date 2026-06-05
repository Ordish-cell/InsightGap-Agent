import hashlib

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem


class SerpApiSource(FeedSource):
    name = "serpapi_source"
    source_type = "web"

    def __init__(self):
        self.enabled = settings.feed_serpapi_enabled and settings.feed_source_serpapi_enabled and bool(settings.serpapi_api_key)
        self.max_items = settings.feed_serpapi_max_items

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                "https://serpapi.com/search.json",
                params={"engine": settings.feed_serpapi_engine, "api_key": settings.serpapi_api_key, "q": "latest AI agent tools", "num": self.max_items, "location": settings.feed_serpapi_location or None, "hl": settings.feed_serpapi_hl, "gl": settings.feed_serpapi_gl},
            )
            response.raise_for_status()
        return [
            RawFeedItem(source_id="serpapi:" + hashlib.sha256(item.get("link", "").encode()).hexdigest(), source_type="web", title=item.get("title", ""), summary=item.get("snippet", ""), url=item.get("link"), raw=item, tags=["serpapi", "web_search"], domain_hints=[])
            for item in response.json().get("organic_results", [])[: self.max_items]
            if item.get("title")
        ]

    def health(self) -> dict:
        if not settings.serpapi_api_key:
            return {"enabled": False, "status": "disabled", "source_type": self.source_type}
        return super().health()
