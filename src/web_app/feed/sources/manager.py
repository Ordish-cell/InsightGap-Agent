import asyncio
import logging
from typing import Any

from src.web_app.core.config import settings
from src.web_app.feed.sources.arxiv import ARXIV_BUCKET_QUERIES, ArxivSource
from src.web_app.feed.sources.base import FeedSource, RawFeedItem
from src.web_app.feed.sources.bucket_seed import BucketSeedSource
from src.web_app.feed.sources.duckduckgo import DUCKDUCKGO_BUCKET_QUERIES, DuckDuckGoSource
from src.web_app.feed.sources.github import GITHUB_BUCKET_LANGUAGES, GITHUB_BUCKET_TOPICS, GitHubSource
from src.web_app.feed.sources.manual_seed import ManualSeedSource
from src.web_app.feed.sources.rss import RSSSource
from src.web_app.feed.sources.serpapi import SERPAPI_BUCKET_QUERIES, SerpApiSource
from src.web_app.feed.sources.tavily import TAVILY_BUCKET_QUERIES, TavilySource

logger = logging.getLogger(__name__)

BUCKET_ORDER = ["explicit_related", "adjacent_domain", "far_domain"]

# Minimum real-search candidates needed per bucket before seed fallback kicks in
_BUCKET_MIN_REAL = {
    "explicit_related": 0,  # never used — BucketSeedSource doesn't cover explicit
    "adjacent_domain": 2,
    "far_domain": 1,
}


def _tavily_enabled() -> bool:
    return (
        settings.feed_tavily_enabled
        and settings.feed_source_tavily_enabled
        and bool(settings.tavily_api_key)
    )


def _serpapi_enabled() -> bool:
    return (
        settings.feed_serpapi_enabled
        and settings.feed_source_serpapi_enabled
        and bool(settings.serpapi_api_key)
    )


def _rss_enabled() -> bool:
    return settings.feed_rss_enabled and bool(settings.csv(settings.feed_rss_urls))


class SearchSourceManager:
    def __init__(self, sources: list[FeedSource] | None = None):
        self.sources = sources or [
            ManualSeedSource(), GitHubSource(), ArxivSource(),
            RSSSource(), TavilySource(), SerpApiSource(), DuckDuckGoSource(),
        ]

    def get_enabled_sources(self) -> list[FeedSource]:
        return [source for source in self.sources if source.enabled]

    def health(self) -> dict[str, Any]:
        return {source.name: source.health() for source in self.sources}

    async def fetch_all(self) -> tuple[list[RawFeedItem], dict[str, Any], dict[str, dict]]:
        """Two-phase fetch per bucket: real search first, BucketSeed fallback only when short.

        Returns (items, stats, source_summary).
        source_summary = {
            "explicit_related": {"search_count": N, "seed_count": 0, "providers": [...]},
            ...
        }
        """
        items: list[RawFeedItem] = []
        stats: dict[str, Any] = {}
        source_summary: dict[str, dict] = {}

        for bucket in BUCKET_ORDER:
            real_sources = self._build_real_sources(bucket)
            bucket_providers_list = [s.name.replace("_source", "") for s in real_sources if s.enabled]
            logger.info("bucket sources bucket=%s providers=%s", bucket, bucket_providers_list)
            bucket_items: list[RawFeedItem] = []
            real_count = 0
            bucket_providers: list[str] = []

            # Phase 1: Real search sources
            for source in real_sources:
                if not source.enabled:
                    continue
                provider = source.name.replace("_source", "")
                queries = self._source_queries(source, bucket)
                query_desc = ", ".join(queries[:3]) if queries else "default"
                try:
                    rows = await asyncio.wait_for(source.fetch(), timeout=15)
                    for row in rows:
                        row.search_bucket = bucket
                    bucket_items.extend(rows)
                    real_count += len(rows)
                    bucket_providers.append(provider)
                    logger.info(
                        "feed source fetch bucket=%s provider=%s source_kind=search query=%s returned=%d",
                        bucket, provider, query_desc[:120], len(rows),
                    )
                except Exception as exc:
                    label = f"{bucket}/{source.name}"
                    stats[label] = {"enabled": True, "fetched": 0, "failed": True, "status": "degraded", "error": str(exc)[:200]}
                    logger.warning("feed source fetch bucket=%s provider=%s failed: %s", bucket, provider, str(exc)[:120])

            # Phase 2: BucketSeedSource fallback for adjacent/far when real search is short
            seed_count = 0
            min_needed = _BUCKET_MIN_REAL.get(bucket, 0)
            if bucket in ("adjacent_domain", "far_domain") and real_count < min_needed:
                seed_source = BucketSeedSource(buckets=[bucket])
                try:
                    seed_rows = await asyncio.wait_for(seed_source.fetch(), timeout=5)
                    for row in seed_rows:
                        row.search_bucket = bucket
                    bucket_items.extend(seed_rows)
                    seed_count = len(seed_rows)
                    if seed_count:
                        bucket_providers.append("bucket_seed")
                        logger.info(
                            "feed source fetch bucket=%s provider=bucket_seed source_kind=bucket_seed returned=%d",
                            bucket, seed_count,
                        )
                except Exception as exc:
                    logger.warning("feed bucket seed fallback failed bucket=%s: %s", bucket, str(exc)[:120])

            items.extend(bucket_items)
            source_summary[bucket] = {
                "search_count": real_count,
                "seed_count": seed_count,
                "providers": bucket_providers,
            }
            logger.info(
                "feed bucket fill bucket=%s real_candidates=%d seed_candidates=%d selected=%d",
                bucket, real_count, seed_count, len(bucket_items),
            )

        # Global fallback: if nothing at all, use ManualSeedSource
        if not items:
            fallback = ManualSeedSource()
            rows = await fallback.fetch()
            for row in rows:
                row.search_bucket = "explicit_related"
            items.extend(rows)
            stats[fallback.name] = {"enabled": True, "fetched": len(rows), "failed": False, "status": "ok", "error": None}
            source_summary["explicit_related"] = {"search_count": 0, "seed_count": len(rows), "providers": ["manual_seed"]}

        return items, stats, source_summary

    def _build_real_sources(self, bucket: str) -> list[FeedSource]:
        """Build the list of real search sources for a bucket (no BucketSeedSource)."""
        sources: list[FeedSource] = []

        # DuckDuckGo — always, for all buckets
        dq = DUCKDUCKGO_BUCKET_QUERIES.get(bucket, [])
        if dq:
            sources.append(DuckDuckGoSource(queries=dq))

        # Tavily — if API key configured
        if _tavily_enabled():
            tq = TAVILY_BUCKET_QUERIES.get(bucket, [])
            if tq:
                sources.append(TavilySource(queries=tq[:3]))

        # SerpAPI — if API key configured
        if _serpapi_enabled():
            sq = SERPAPI_BUCKET_QUERIES.get(bucket, [])
            if sq:
                sources.append(SerpApiSource(queries=sq[:3]))

        # RSS — if feed URLs configured
        if _rss_enabled():
            sources.append(RSSSource())

        # GitHub — explicit + adjacent only, NEVER far_domain
        if bucket != "far_domain":
            gt = GITHUB_BUCKET_TOPICS.get(bucket, [])
            gl = GITHUB_BUCKET_LANGUAGES.get(bucket, [])
            if gt:
                sources.append(GitHubSource(topics=gt[:3], languages=gl[:2]))

        # Arxiv — explicit + adjacent only
        if bucket != "far_domain":
            aq = ARXIV_BUCKET_QUERIES.get(bucket, [])
            if aq:
                sources.append(ArxivSource(queries=aq[:4]))

        return sources

    @staticmethod
    def _source_queries(source: FeedSource, bucket: str) -> list[str]:
        """Extract queries from a source for logging."""
        queries = getattr(source, "_queries", None)
        if queries:
            return queries
        topics = getattr(source, "_topics", None)
        if topics:
            return topics
        languages = getattr(source, "_languages", None)
        if languages:
            return languages
        return []
