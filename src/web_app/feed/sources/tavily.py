import hashlib

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem


TAVILY_BUCKET_QUERIES: dict[str, list[str]] = {
    "explicit_related": [
        "LangGraph agent memory RAG MCP",
        "LLM agent framework tool use memory",
    ],
    "adjacent_domain": [
        "LLM observability tracing monitoring",
        "AI workflow automation n8n Temporal",
        "human in the loop AI approval workflow",
        "browser automation Playwright AI workflow",
    ],
    "far_domain": [
        "product analytics user feedback opportunity signals",
        "competitive intelligence market signal monitoring",
        "education AI adaptive learning feedback loop",
        "investment research alternative data early market signals",
    ],
}


class TavilySource(FeedSource):
    name = "tavily_source"
    source_type = "web"

    def __init__(self, queries: list[str] | None = None):
        self.enabled = settings.feed_tavily_enabled and settings.feed_source_tavily_enabled and bool(settings.tavily_api_key)
        self.max_items = settings.feed_tavily_max_items
        self._queries = queries

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        queries = self._queries if self._queries else ["latest AI agent frameworks LangGraph RAG MCP"]
        items: list[RawFeedItem] = []
        async with httpx.AsyncClient(timeout=10) as client:
            for q in queries:
                if len(items) >= self.max_items:
                    break
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": settings.tavily_api_key, "query": q, "search_depth": settings.feed_tavily_search_depth, "max_results": max(3, self.max_items // len(queries)), "include_answer": settings.feed_tavily_include_answer, "include_raw_content": settings.feed_tavily_include_raw_content},
                )
                response.raise_for_status()
                for item in response.json().get("results", []):
                    if not item.get("title"):
                        continue
                    items.append(RawFeedItem(
                        source_id="tavily:" + hashlib.sha256(item.get("url", "").encode()).hexdigest(),
                        source_type="web", title=item.get("title", ""),
                        summary=item.get("content", ""), url=item.get("url"), raw=item,
                        tags=["tavily", "web_search"], domain_hints=[],
                        provider="tavily", source_kind="search", search_query=q,
                    ))
        return items

    def health(self) -> dict:
        if not settings.tavily_api_key:
            return {"enabled": False, "status": "disabled", "source_type": self.source_type}
        return super().health()
