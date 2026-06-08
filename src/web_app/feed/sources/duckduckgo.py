import hashlib

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem


DUCKDUCKGO_BUCKET_QUERIES: dict[str, list[str]] = {
    "explicit_related": [
        "LangGraph agent memory RAG MCP",
        "AI agent OS context engineering",
        "LLM agent framework tool use memory",
        "multi-agent workflow LangGraph",
        "RAG evaluation agent system",
    ],
    "adjacent_domain": [
        "AI workflow automation n8n Temporal",
        "LLM observability tracing monitoring",
        "human in the loop AI approval workflow",
        "developer productivity AI coding assistant",
        "personal knowledge management AI knowledge graph",
        "browser automation Playwright AI",
        "context engineering prompt management",
        "AI evaluation observability product",
    ],
    "far_domain": [
        "AI product analytics user feedback loop",
        "startup market intelligence signal detection",
        "competitive intelligence product discovery",
        "enterprise knowledge management workflow",
        "education AI adaptive learning product",
        "investment research alternative data AI",
        "knowledge operations automation enterprise",
        "product-led growth AI analytics",
    ],
}


class DuckDuckGoSource(FeedSource):
    name = "duckduckgo_source"
    source_type = "web"

    def __init__(self, queries: list[str] | None = None):
        self.enabled = settings.feed_duckduckgo_enabled and settings.feed_source_duckduckgo_enabled
        self.max_items = settings.feed_duckduckgo_max_items
        self._queries = queries

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        queries = self._queries if self._queries else ["latest LangGraph Agent RAG MCP"]
        items: list[RawFeedItem] = []
        try:
            from duckduckgo_search import DDGS

            per_query = max(3, self.max_items // len(queries))
            for q in queries:
                if len(items) >= self.max_items:
                    break
                rows = DDGS().text(q, region=settings.feed_duckduckgo_region, safesearch=settings.feed_duckduckgo_safesearch, timelimit=settings.feed_duckduckgo_time, max_results=per_query)
                for row in rows:
                    if not row.get("title"):
                        continue
                    source_id = "duckduckgo:" + hashlib.sha256(row.get("href", "").encode()).hexdigest()
                    items.append(RawFeedItem(source_id=source_id, source_type="web", title=row.get("title", ""), summary=row.get("body", ""), url=row.get("href"), raw=row, tags=["duckduckgo", "web_search"], domain_hints=[]))
        except Exception:
            pass
        return items
