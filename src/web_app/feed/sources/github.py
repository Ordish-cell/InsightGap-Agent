from datetime import date, timedelta
from typing import Any

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem, parse_datetime


class GitHubSource(FeedSource):
    name = "github_source"
    source_type = "github"

    def __init__(self):
        self.enabled = settings.feed_github_enabled
        self.max_items = settings.feed_github_max_items

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        pushed_after = (date.today() - timedelta(days=settings.feed_github_pushed_days)).isoformat()
        queries = [f"topic:{topic} stars:>{settings.feed_github_min_stars} pushed:>{pushed_after}" for topic in settings.csv(settings.feed_github_topics)]
        queries += [f"language:{lang} agent stars:>{settings.feed_github_min_stars} pushed:>{pushed_after}" for lang in settings.csv(settings.feed_github_languages)]
        items: list[RawFeedItem] = []
        headers = {"Accept": "application/vnd.github+json"}
        if settings.github_token:
            headers["Authorization"] = "Bearer " + settings.github_token
        async with httpx.AsyncClient(timeout=8) as client:
            for query in queries:
                if len(items) >= self.max_items:
                    break
                response = await client.get("https://api.github.com/search/repositories", params={"q": query, "sort": "updated", "order": "desc", "per_page": min(10, self.max_items)}, headers=headers)
                response.raise_for_status()
                for repo in response.json().get("items", []):
                    items.append(self._from_repo(repo))
                    if len(items) >= self.max_items:
                        break
        return items

    def _from_repo(self, repo: dict[str, Any]) -> RawFeedItem:
        topics = repo.get("topics") or []
        return RawFeedItem(
            source_id=f"github:{repo.get('full_name')}",
            source_type="github",
            title=repo.get("full_name") or repo.get("name") or "",
            summary=repo.get("description") or "",
            url=repo.get("html_url"),
            published_at=parse_datetime(repo.get("pushed_at") or repo.get("updated_at")),
            author=(repo.get("owner") or {}).get("login"),
            raw=repo,
            tags=["github", "repo", *(topics or [])],
            domain_hints=["ai", "agent", "devtools"],
        )
