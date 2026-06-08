import xml.etree.ElementTree as ET

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem, parse_datetime


ARXIV_BUCKET_QUERIES: dict[str, list[str]] = {
    "explicit_related": [
        "agent memory RAG MCP",
        "LLM agent tool use multi-agent",
        "LangGraph workflow agent",
        "retrieval augmented generation agent",
        "AI agent context engineering",
    ],
    "adjacent_domain": [
        "workflow automation AI orchestration",
        "LLM observability monitoring tracing",
        "human-in-the-loop AI approval",
        "developer productivity coding assistant",
        "knowledge graph personal knowledge management",
        "browser automation web agent",
        "prompt management context engineering",
        "AI evaluation benchmark system",
    ],
    "far_domain": [],
}


class ArxivSource(FeedSource):
    name = "arxiv_source"
    source_type = "paper"

    def __init__(self, queries: list[str] | None = None):
        self.enabled = settings.feed_arxiv_enabled and settings.feed_source_arxiv_enabled
        self.max_items = settings.feed_arxiv_max_items
        self._queries = queries

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        terms = self._queries if self._queries else settings.csv(settings.feed_arxiv_queries)[:4]
        query = " OR ".join(f'all:"{item}"' for item in terms)
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get("https://export.arxiv.org/api/query", params={"search_query": query, "start": 0, "max_results": self.max_items, "sortBy": "submittedDate", "sortOrder": "descending"})
            response.raise_for_status()
        return self._parse(response.text)

    def _parse(self, xml_text: str) -> list[RawFeedItem]:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        items = []
        for entry in root.findall("atom:entry", ns):
            entry_id = (entry.findtext("atom:id", "", ns) or "").rsplit("/", 1)[-1]
            title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
            summary = " ".join((entry.findtext("atom:summary", "", ns) or "").split())
            authors = [author.findtext("atom:name", "", ns) for author in entry.findall("atom:author", ns)]
            categories = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ns)]
            items.append(
                RawFeedItem(
                    source_id=f"arxiv:{entry_id}",
                    source_type="paper",
                    title=title,
                    summary=summary,
                    url=f"https://arxiv.org/abs/{entry_id}",
                    published_at=parse_datetime(entry.findtext("atom:published", "", ns)),
                    author=", ".join([item for item in authors if item]),
                    raw={"categories": categories},
                    tags=["paper", "arxiv", *categories],
                    domain_hints=["ai", "research"],
                    provider="arxiv",
                    source_kind="search",
                    search_query=query,
                )
            )
        return items
