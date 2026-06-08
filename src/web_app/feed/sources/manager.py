import asyncio
import logging
from typing import Any

from src.web_app.feed.sources.arxiv import ARXIV_BUCKET_QUERIES, ArxivSource
from src.web_app.feed.sources.base import FeedSource, RawFeedItem
from src.web_app.feed.sources.bucket_seed import BucketSeedSource
from src.web_app.feed.sources.duckduckgo import DUCKDUCKGO_BUCKET_QUERIES, DuckDuckGoSource
from src.web_app.feed.sources.github import GITHUB_BUCKET_LANGUAGES, GITHUB_BUCKET_TOPICS, GitHubSource
from src.web_app.feed.sources.manual_seed import ManualSeedSource
from src.web_app.feed.sources.rss import RSSSource
from src.web_app.feed.sources.serpapi import SerpApiSource
from src.web_app.feed.sources.tavily import TavilySource

logger = logging.getLogger(__name__)

BUCKET_ORDER = ["explicit_related", "adjacent_domain", "far_domain"]


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
        bucket_counts: dict[str, int] = {}

        for bucket in BUCKET_ORDER:
            bucket_sources = self._build_bucket_sources(bucket)
            for source in bucket_sources:
                if not source.enabled:
                    continue
                query_desc = getattr(source, "_queries", getattr(source, "_topics", "default"))
                logger.info("feed source bucket=%s source=%s query=%s", bucket, source.name, str(query_desc)[:120])
                try:
                    rows = await asyncio.wait_for(source.fetch(), timeout=15)
                    for row in rows:
                        row.search_bucket = bucket
                    items.extend(rows)
                    label = f"{bucket}/{source.name}"
                    stats[label] = {"enabled": True, "fetched": len(rows), "failed": False, "status": "ok", "error": None}
                    logger.info("feed source bucket=%s source=%s returned=%d", bucket, source.name, len(rows))
                except Exception as exc:
                    label = f"{bucket}/{source.name}"
                    stats[label] = {"enabled": True, "fetched": 0, "failed": True, "status": "degraded", "error": str(exc)[:200]}

            bucket_counts[bucket] = sum(1 for item in items if item.search_bucket == bucket)

        for bucket in BUCKET_ORDER:
            count = bucket_counts.get(bucket, 0)
            logger.info("feed bucket %s: %d items", bucket, count)

        if not items:
            fallback = ManualSeedSource()
            rows = await fallback.fetch()
            for row in rows:
                row.search_bucket = "explicit_related"
            items.extend(rows)
            stats[fallback.name] = {"enabled": True, "fetched": len(rows), "failed": False, "status": "ok", "error": None}

        return items, stats

    def _build_bucket_sources(self, bucket: str) -> list[FeedSource]:
        sources: list[FeedSource] = []

        dq = DUCKDUCKGO_BUCKET_QUERIES.get(bucket, [])
        if dq:
            sources.append(DuckDuckGoSource(queries=dq))

        aq = ARXIV_BUCKET_QUERIES.get(bucket, [])
        if aq:
            sources.append(ArxivSource(queries=aq[:4]))

        gt = GITHUB_BUCKET_TOPICS.get(bucket, [])
        gl = GITHUB_BUCKET_LANGUAGES.get(bucket, [])
        if gt:
            sources.append(GitHubSource(topics=gt[:3], languages=gl[:2]))

        # BucketSeedSource as guaranteed fallback for adjacent/far
        if bucket in ("adjacent_domain", "far_domain"):
            sources.append(BucketSeedSource(buckets=[bucket]))

        return sources
