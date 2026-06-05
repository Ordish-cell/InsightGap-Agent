import xml.etree.ElementTree as ET

import httpx

from src.web_app.core.config import settings
from src.web_app.feed.sources.base import FeedSource, RawFeedItem, parse_datetime


class ArxivSource(FeedSource):
    name = "arxiv_source"
    source_type = "paper"

    def __init__(self):
        self.enabled = settings.feed_arxiv_enabled and settings.feed_source_arxiv_enabled
        self.max_items = settings.feed_arxiv_max_items

    async def fetch(self) -> list[RawFeedItem]:
        if not self.enabled:
            return []
        query = " OR ".join(f'all:"{item}"' for item in settings.csv(settings.feed_arxiv_queries)[:4])
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
                )
            )
        return items
