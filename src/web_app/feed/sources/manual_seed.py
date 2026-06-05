import json
from pathlib import Path

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem, parse_datetime

DEFAULT_SEEDS = [
    {
        "source_id": "seed_agent_001",
        "source_type": "manual",
        "title": "LangGraph is a practical foundation for multi-agent runtimes",
        "summary": "LangGraph provides graph-based orchestration primitives for stateful, observable agent workflows.",
        "url": "https://github.com/langchain-ai/langgraph",
        "published_at": "2026-06-01T00:00:00",
        "tags": ["agent", "langgraph", "multi-agent"],
        "domain_hints": ["ai", "agent", "devtools"],
    },
    {
        "source_id": "seed_rag_001",
        "source_type": "manual",
        "title": "RAGAS tracks retrieval and generation quality for RAG systems",
        "summary": "RAG evaluation tooling is becoming important for evidence-aware knowledge agents.",
        "url": "https://github.com/explodinggradients/ragas",
        "published_at": "2026-06-01T00:00:00",
        "tags": ["rag", "evaluation", "agent"],
        "domain_hints": ["ai", "research", "product"],
    },
    {
        "source_id": "seed_startup_001",
        "source_type": "manual",
        "title": "MCP standardizes tool context for agent systems",
        "summary": "The Model Context Protocol creates a reusable integration layer for agent tools and data sources.",
        "url": "https://modelcontextprotocol.io/",
        "published_at": "2026-06-01T00:00:00",
        "tags": ["browser agent", "startup", "automation"],
        "domain_hints": ["startup", "automation", "agent"],
    },
]


class ManualSeedSource(FeedSource):
    name = "manual_seed_source"
    source_type = "manual"

    def __init__(self):
        self.enabled = settings.feed_manual_seed_enabled and settings.feed_source_manual_seed_enabled
        self.max_items = settings.feed_refresh_max_items

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        path = Path("config/feed_sources_seed.json")
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else DEFAULT_SEEDS
        return [
            RawFeedItem(
                source_id=row["source_id"],
                source_type=row.get("source_type", "manual"),
                title=row["title"],
                summary=row.get("summary", ""),
                url=row.get("url"),
                published_at=parse_datetime(row.get("published_at")),
                author=row.get("author"),
                raw=row,
                tags=row.get("tags", []),
                domain_hints=row.get("domain_hints", []),
            )
            for row in rows[: self.max_items]
        ]
