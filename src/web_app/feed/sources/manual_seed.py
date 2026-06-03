import json
from pathlib import Path

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem, parse_datetime

DEFAULT_SEEDS = [
    {
        "source_id": "seed_agent_001",
        "source_type": "manual",
        "title": "LangGraph multi-agent runtime patterns are worth watching",
        "summary": "Agent orchestration is moving from simple ReAct loops toward recoverable, observable, evaluable graph runtimes.",
        "url": "https://example.com/langgraph-agent-runtime",
        "published_at": "2026-06-01T00:00:00",
        "tags": ["agent", "langgraph", "multi-agent"],
        "domain_hints": ["ai", "agent", "devtools"],
    },
    {
        "source_id": "seed_rag_001",
        "source_type": "manual",
        "title": "RAG evaluation is becoming a product advantage",
        "summary": "Teams that track retrieval evidence, faithfulness, and actionability can ship safer knowledge agents.",
        "url": "https://example.com/rag-evaluation-product",
        "published_at": "2026-06-01T00:00:00",
        "tags": ["rag", "evaluation", "agent"],
        "domain_hints": ["ai", "research", "product"],
    },
    {
        "source_id": "seed_startup_001",
        "source_type": "manual",
        "title": "Browser agents may create new workflow automation startups",
        "summary": "The opportunity is less about demos and more about approval gates, replayability, and domain-specific workflows.",
        "url": "https://example.com/browser-agent-startups",
        "published_at": "2026-06-01T00:00:00",
        "tags": ["browser agent", "startup", "automation"],
        "domain_hints": ["startup", "automation", "agent"],
    },
]


class ManualSeedSource(FeedSource):
    name = "manual_seed_source"
    source_type = "manual"

    def __init__(self):
        self.enabled = settings.feed_manual_seed_enabled
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
