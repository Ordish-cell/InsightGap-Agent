from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from src.web_app.core.config import settings


WEB_SEARCH_TOOL_ALIASES = {
    "web.search": [
        "web_search",
        "internet_search",
        "search_web",
        "联网搜索",
        "上网搜索",
        "上网查",
        "查最新",
    ],
}


class WebSearchProvider:
    """Small synchronous provider for the read-only web.search MCP tool."""

    def search(self, query: str, *, limit: int = 5, recency_days: int | None = None) -> dict[str, Any]:
        original_query = (query or "").strip()
        limit = max(1, min(int(limit or 5), 10))
        recency_days = self._safe_int(recency_days)
        if not original_query:
            return self._failure(original_query, "empty_query")

        rounds: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        attempted: set[str] = set()
        query = original_query
        for round_index in range(1, 3):
            if query in attempted:
                break
            attempted.add(query)
            result = self._search_once(query, limit, recency_days)
            results = result.get("results") if isinstance(result.get("results"), list) else []
            observation = self._observe_results(query, results, result.get("error", ""))
            result["_react_observation"] = observation
            rounds.append({
                "round": round_index,
                "query": query,
                "provider": result.get("provider") or "",
                "result_count": len(results),
                "observation": observation,
                "error": result.get("error") or "",
            })
            if best is None or self._result_quality(results) > self._result_quality(best.get("results") or []):
                best = result
            if self._is_good_enough(original_query, results):
                break
            refined = self._refine_query(original_query, query, results)
            if not refined or refined == query:
                break
            query = refined

        final = best or self._failure(original_query, "no_results")
        executed_query = str(final.get("query") or query)
        final["query"] = original_query
        final["final_query"] = executed_query
        final["search_rounds"] = rounds
        final["reasoning_summary"] = self._reasoning_summary(rounds)
        return final

    def _search_once(self, query: str, limit: int, recency_days: int | None) -> dict[str, Any]:
        errors: list[str] = []
        if settings.tavily_api_key:
            tavily = self._search_tavily(query, limit, recency_days)
            if tavily.get("results"):
                return tavily
            if tavily.get("error"):
                errors.append(f"tavily: {tavily['error']}")

        if settings.serpapi_api_key:
            serpapi = self._search_serpapi(query, limit, recency_days)
            serpapi["used_fallback"] = bool(settings.tavily_api_key)
            if serpapi.get("results"):
                return serpapi
            if serpapi.get("error"):
                errors.append(f"serpapi: {serpapi['error']}")

        if not settings.tavily_api_key and not settings.serpapi_api_key:
            errors.append("no_provider_configured")
        return self._failure(query, "; ".join(errors) or "no_results")

    def _observe_results(self, query: str, results: list[dict[str, Any]], error: str) -> str:
        if not results:
            return f"未读取到有效搜索结果：{error or 'no_results'}"
        official_count = sum(1 for item in results if self._is_officialish_result(item))
        if official_count:
            return f"读取到 {len(results)} 条结果，其中 {official_count} 条来自较权威/官方来源。"
        return f"读取到 {len(results)} 条结果，但未明显命中官方来源，必要时改写查询继续验证。"

    def _is_good_enough(self, original_query: str, results: list[dict[str, Any]]) -> bool:
        if not results:
            return False
        normalized = original_query.lower()
        compact = "".join(normalized.split())
        needs_authority = any(term in normalized for term in ("latest", "current", "release", "version", "model", "price", "stock")) or any(
            term in compact for term in ("最新", "当前", "发布", "版本", "模型", "价格", "股价")
        )
        if not needs_authority:
            return len(results) >= 2
        return any(self._is_officialish_result(item) for item in results)

    def _result_quality(self, results: list[dict[str, Any]]) -> int:
        return len(results) + 3 * sum(1 for item in results if self._is_officialish_result(item))

    def _is_officialish_result(self, item: dict[str, Any]) -> bool:
        url = str(item.get("url") or "").lower()
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        official_markers = (
            ".com", ".org", ".ai", ".dev", ".io", ".gov", ".edu",
        )
        known_good = (
            "anthropic.com", "docs.anthropic.com", "openai.com", "docs.openai.com",
            "github.com", "microsoft.com", "google.com", "cloud.google.com",
            "aws.amazon.com", "docs.aws.amazon.com", "nextjs.org", "react.dev",
            "python.org", "pypi.org", "npmjs.com", "nodejs.org",
        )
        bad_hosts = ("wikipedia.org", "linkedin.com", "youtube.com", "x.com", "twitter.com", "apps.apple.com", "play.google.com")
        if any(bad in host for bad in bad_hosts):
            return False
        if any(good in host for good in known_good):
            return True
        title = str(item.get("title") or "").lower()
        return "official" in title and any(marker in host for marker in official_markers)

    def _refine_query(self, original_query: str, current_query: str, results: list[dict[str, Any]]) -> str:
        normalized = original_query.lower()
        compact = "".join(normalized.split())
        asks_latest = any(term in normalized for term in ("latest", "current", "recent")) or any(term in compact for term in ("最新", "当前"))
        asks_model = any(term in normalized for term in ("model", "models")) or "模型" in compact
        asks_version = any(term in normalized for term in ("version", "release")) or any(term in compact for term in ("版本", "发布"))
        if "claude" in normalized and asks_latest and asks_model:
            if "anthropic.com/news" not in current_query:
                return "site:anthropic.com/news Claude Opus latest model"
            return "site:docs.anthropic.com Claude models Opus latest"
        if asks_latest and asks_model:
            subject = self._extract_subject(original_query, ("最新模型", "当前模型", "latest model", "current model"))
            if subject:
                return f"{subject} latest model official announcement docs"
        if asks_latest and asks_version:
            subject = self._extract_subject(original_query, ("最新版本", "当前版本", "latest version", "current version"))
            if subject:
                return f"{subject} latest version official docs release notes"
        if not any(self._is_officialish_result(item) for item in results):
            return f"{original_query} official source"
        return ""

    def _extract_subject(self, query: str, markers: tuple[str, ...]) -> str:
        lower = query.lower()
        for marker in markers:
            marker_lower = marker.lower()
            index = lower.find(marker_lower)
            if index > 0:
                return query[:index].strip(" ：:，,。?？的")
        return ""

    def _reasoning_summary(self, rounds: list[dict[str, Any]]) -> str:
        if not rounds:
            return "未执行联网搜索。"
        if len(rounds) == 1:
            return rounds[0].get("observation") or "完成 1 轮联网搜索。"
        return "；".join(str(item.get("observation") or "") for item in rounds if item.get("observation"))

    def _search_tavily(self, query: str, limit: int, recency_days: int | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": settings.feed_tavily_search_depth or "basic",
            "max_results": limit,
            "include_answer": False,
            "include_raw_content": False,
        }
        if recency_days:
            payload["days"] = max(1, int(recency_days))
        try:
            with httpx.Client(timeout=12) as client:
                response = client.post("https://api.tavily.com/search", json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return self._failure(query, str(exc), provider="tavily")

        results = []
        for item in data.get("results", [])[:limit]:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title and not url:
                continue
            results.append({
                "title": title or url,
                "url": url,
                "snippet": str(item.get("content") or "").strip(),
                "provider": "tavily",
                "published_at": item.get("published_date") or item.get("published_at"),
                "score": item.get("score"),
            })
        return {"query": query, "provider": "tavily", "used_fallback": False, "results": results, "error": "" if results else "no_results"}

    def _search_serpapi(self, query: str, limit: int, recency_days: int | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": settings.feed_serpapi_engine or "google",
            "api_key": settings.serpapi_api_key,
            "q": query,
            "num": limit,
            "location": settings.feed_serpapi_location or None,
            "hl": settings.feed_serpapi_hl,
            "gl": settings.feed_serpapi_gl,
        }
        tbs = self._serpapi_tbs(recency_days)
        if tbs:
            params["tbs"] = tbs
        try:
            with httpx.Client(timeout=12) as client:
                response = client.get("https://serpapi.com/search.json", params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return self._failure(query, str(exc), provider="serpapi")

        results = []
        for item in data.get("organic_results", [])[:limit]:
            title = str(item.get("title") or "").strip()
            url = str(item.get("link") or "").strip()
            if not title and not url:
                continue
            results.append({
                "title": title or url,
                "url": url,
                "snippet": str(item.get("snippet") or "").strip(),
                "provider": "serpapi",
                "published_at": item.get("date"),
                "score": None,
            })
        return {"query": query, "provider": "serpapi", "used_fallback": False, "results": results, "error": "" if results else "no_results"}

    @staticmethod
    def _serpapi_tbs(recency_days: int | None) -> str:
        if not recency_days:
            return ""
        days = max(1, int(recency_days))
        if days <= 1:
            return "qdr:d"
        if days <= 7:
            return "qdr:w"
        if days <= 31:
            return "qdr:m"
        return "qdr:y"

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _failure(query: str, error: str, *, provider: str = "") -> dict[str, Any]:
        return {"query": query, "provider": provider, "used_fallback": False, "results": [], "error": str(error or "search_failed")}


web_search_provider = WebSearchProvider()
