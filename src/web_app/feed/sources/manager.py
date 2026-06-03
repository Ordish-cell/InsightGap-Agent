import asyncio
from typing import Any

from src.web_app.feed.sources.arxiv import ArxivSource
from src.web_app.feed.sources.base import FeedSource, RawFeedItem
from src.web_app.feed.sources.duckduckgo import DuckDuckGoSource
from src.web_app.feed.sources.github import GitHubSource
from src.web_app.feed.sources.manual_seed import ManualSeedSource
from src.web_app.feed.sources.rss import RSSSource
from src.web_app.feed.sources.serpapi import SerpApiSource
from src.web_app.feed.sources.tavily import TavilySource


class SearchSourceManager:
    def __init__(self, sources: list[FeedSource] | None = None):
        self.sources = sources or [ManualSeedSource(), GitHubSource(), ArxivSource(), RSSSource(), TavilySource(), SerpApiSource(), DuckDuckGoSource()]

    def get_enabled_sources(self) -> list[FeedSource]:
        return [source for source in self.sources if source.enabled]

    def health(self) -> dict[str, Any]:
        return {source.name: source.health() for source in self.sources}

    async def fetch_all(self) -> tuple[list[RawFeedItem], dict[str, Any]]:
        items: list[RawFeedItem] = []
        stats: dict[str, Any] = {}
        for source in self.sources:
            if not source.enabled:
                stats[source.name] = {"enabled": False, "fetched": 0, "failed": False, "status": "disabled", "error": None}
                continue
            try:
                rows = await asyncio.wait_for(source.fetch(), timeout=15)
                items.extend(rows)
                stats[source.name] = {"enabled": True, "fetched": len(rows), "failed": False, "status": "ok", "error": None}
            except Exception as exc:
                stats[source.name] = {"enabled": True, "fetched": 0, "failed": True, "status": "degraded", "error": str(exc)[:200]}
        if not items:
            fallback = ManualSeedSource()
            rows = await fallback.fetch()
            items.extend(rows)
            stats[fallback.name] = {"enabled": True, "fetched": len(rows), "failed": False, "status": "ok", "error": None}
        return items, stats
