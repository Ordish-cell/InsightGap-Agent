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
        "LLM observability tracing monitoring",
        "AI workflow automation n8n Temporal",
        "human in the loop AI approval workflow",
        "browser automation Playwright AI workflow",
        "developer productivity AI coding assistant",
        "prompt management context engineering evaluation",
        "personal knowledge management AI knowledge graph",
    ],
    "far_domain": [
        "product analytics user feedback opportunity signals",
        "competitive intelligence market signal monitoring",
        "education AI adaptive learning feedback loop",
        "investment research alternative data early market signals",
        "enterprise knowledge operations continuous workflow",
        "product-led growth behavioral analytics expansion opportunity",
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
                    items.append(RawFeedItem(
                        source_id=source_id, source_type="web",
                        title=row.get("title", ""), summary=row.get("body", ""),
                        url=row.get("href"), raw=row,
                        tags=["duckduckgo", "web_search"], domain_hints=[],
                        provider="duckduckgo", source_kind="search", search_query=q,
                    ))
        except Exception:
            pass
        return items
