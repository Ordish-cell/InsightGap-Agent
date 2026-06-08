import hashlib

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem


SERPAPI_BUCKET_QUERIES: dict[str, list[str]] = {
    "explicit_related": [
        "LangGraph agent memory RAG MCP",
        "LLM agent framework tool use",
    ],
    "adjacent_domain": [
        "LLM observability tracing monitoring",
        "AI workflow automation n8n",
        "human in the loop AI approval",
    ],
    "far_domain": [
        "product analytics user feedback signals",
        "competitive intelligence market monitoring",
        "education AI adaptive learning",
    ],
}


class SerpApiSource(FeedSource):
    name = "serpapi_source"
    source_type = "web"

    def __init__(self, queries: list[str] | None = None):
        self.enabled = settings.feed_serpapi_enabled and settings.feed_source_serpapi_enabled and bool(settings.serpapi_api_key)
        self.max_items = settings.feed_serpapi_max_items
        self._queries = queries

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        queries = self._queries if self._queries else ["latest AI agent tools"]
        items: list[RawFeedItem] = []
        async with httpx.AsyncClient(timeout=10) as client:
            for q in queries:
                if len(items) >= self.max_items:
                    break
                response = await client.get(
                    "https://serpapi.com/search.json",
                    params={"engine": settings.feed_serpapi_engine, "api_key": settings.serpapi_api_key, "q": q, "num": max(3, self.max_items // len(queries)), "location": settings.feed_serpapi_location or None, "hl": settings.feed_serpapi_hl, "gl": settings.feed_serpapi_gl},
                )
                response.raise_for_status()
                for item in response.json().get("organic_results", [])[: self.max_items]:
                    if not item.get("title"):
                        continue
                    items.append(RawFeedItem(
                        source_id="serpapi:" + hashlib.sha256(item.get("link", "").encode()).hexdigest(),
                        source_type="web", title=item.get("title", ""),
                        summary=item.get("snippet", ""), url=item.get("link"), raw=item,
                        tags=["serpapi", "web_search"], domain_hints=[],
                        provider="serpapi", source_kind="search", search_query=q,
                    ))
        return items

    def health(self) -> dict:
        if not settings.serpapi_api_key:
            return {"enabled": False, "status": "disabled", "source_type": self.source_type}
        return super().health()
