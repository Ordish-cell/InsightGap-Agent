import hashlib
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.web_app.feed.sources.base import RawFeedItem


@dataclass
class InfoItemCreate:
    title: str
    summary: str
    content: str
    source_url: str
    source_type: str
    author: str
    published_at: Any
    topics: list[str]
    raw_metadata: dict[str, Any]
    content_hash: str


SOURCE_CREDIBILITY = {"github": 0.75, "paper": 0.85, "blog": 0.78, "news": 0.65, "web": 0.60, "manual": 0.70, "unknown": 0.40}


def normalize_raw_item(raw: RawFeedItem) -> InfoItemCreate | None:
    title = " ".join((raw.title or "").split())
    if not title:
        return None
    summary = " ".join((raw.summary or title).split())
    canonical_url = canonicalize_url(raw.url)
    content_hash = stable_hash(canonical_url or f"{raw.source_type}:{title.lower()}")
    source_type = raw.source_type or "unknown"
    tags = list(dict.fromkeys([*(raw.tags or []), *(raw.domain_hints or [])]))
    return InfoItemCreate(
        title=title[:510],
        summary=summary[:10000],
        content=summary[:10000],
        source_url=(canonical_url or "")[:1000],
        source_type=source_type,
        author=(raw.author or "")[:240],
        published_at=raw.published_at,
        topics=tags,
        raw_metadata={
            "source_id": raw.source_id,
            "canonical_url": canonical_url,
            "raw": raw.raw,
            "tags": tags,
            "domain": infer_domain(tags, title + " " + summary),
            "source_credibility": SOURCE_CREDIBILITY.get(source_type, 0.40),
            "search_bucket": raw.search_bucket,
        },
        content_hash=content_hash,
    )


def canonicalize_url(url: str | None) -> str:
    if not url:
        return ""
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return ""
    # Strip utm_* query params and fragment
    query_parts = parts.query.split("&")
    clean_query = "&".join(p for p in query_parts if p and not p.startswith("utm_"))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), clean_query, ""))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def infer_domain(tags: list[str], text: str) -> str:
    haystack = " ".join(tags).lower() + " " + text.lower()
    for domain, keys in {
        "agent": ["agent", "langgraph", "multi-agent", "browser agent"],
        "rag": ["rag", "retrieval", "qdrant", "vector"],
        "devtools": ["github", "python", "typescript", "framework"],
        "startup": ["startup", "opportunity", "product"],
        "research": ["paper", "arxiv", "benchmark", "eval"],
    }.items():
        if any(key in haystack for key in keys):
            return domain
    return "ai"
