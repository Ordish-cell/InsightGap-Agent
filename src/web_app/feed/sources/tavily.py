import hashlib

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem


class TavilySource(FeedSource):
    name = "tavily_source"
    source_type = "web"

    def __init__(self):
        self.enabled = settings.feed_tavily_enabled and settings.feed_source_tavily_enabled and bool(settings.tavily_api_key)
        self.max_items = settings.feed_tavily_max_items

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        query = "latest AI agent frameworks LangGraph RAG MCP"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": settings.tavily_api_key, "query": query, "search_depth": settings.feed_tavily_search_depth, "max_results": self.max_items, "include_answer": settings.feed_tavily_include_answer, "include_raw_content": settings.feed_tavily_include_raw_content},
            )
            response.raise_for_status()
        return [
            RawFeedItem(source_id="tavily:" + hashlib.sha256(item.get("url", "").encode()).hexdigest(), source_type="web", title=item.get("title", ""), summary=item.get("content", ""), url=item.get("url"), raw=item, tags=["tavily", "web_search"], domain_hints=[])
            for item in response.json().get("results", [])
            if item.get("title")
        ]

    def health(self) -> dict:
        if not settings.tavily_api_key:
            return {"enabled": False, "status": "disabled", "source_type": self.source_type}
        return super().health()
