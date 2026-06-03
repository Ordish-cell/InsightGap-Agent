from datetime import UTC, datetime, timedelta
from typing import Any

from src.web_app.core.config import settings


class FeedScorer:
    def score(self, info_item: Any, user_profile: Any, user_feedback_stats: dict | None = None) -> dict[str, Any]:
        feedback = user_feedback_stats or {}
        topics = [str(item).lower() for item in (info_item.topics or [])]
        text = f"{info_item.title} {info_item.summary} {' '.join(topics)}".lower()
        explicit = _profile_terms(user_profile, "explicit_interests") + _profile_terms(user_profile, "goals")
        adjacent = _profile_terms(user_profile, "adjacent_domains")
        far = _profile_terms(user_profile, "far_domains")

        explicit_hits = [term for term in explicit if term and term in text]
        adjacent_hits = [term for term in adjacent if term and term in text]
        far_hits = [term for term in far if term and term in text]
        if explicit_hits:
            relation_type = "explicit_related"
            relevance = 0.65
        elif adjacent_hits:
            relation_type = "adjacent_domain"
            relevance = 0.45
        elif far_hits:
            relation_type = "far_domain"
            relevance = 0.30
        else:
            relation_type = "far_domain"
            relevance = 0.22 if _has_opportunity_terms(text) else 0.12

        for topic in topics:
            relevance += 0.08 * feedback.get("positive_topics", {}).get(topic, 0)
            relevance -= 0.10 * feedback.get("negative_topics", {}).get(topic, 0)
        relevance = _clamp(relevance)
        credibility = float((info_item.raw_metadata or {}).get("source_credibility", 0.40))
        novelty = _novelty(info_item.published_at)
        cross_domain = {"explicit_related": 0.35, "adjacent_domain": 0.70, "far_domain": 0.82}[relation_type]
        opportunity = 0.75 if _has_opportunity_terms(text) else 0.45
        actionability = 0.75 if any(term in text for term in ["github", "tool", "framework", "workflow", "agent", "rag"]) else 0.50
        final = round(0.30 * relevance + 0.20 * novelty + 0.15 * cross_domain + 0.15 * opportunity + 0.10 * credibility + 0.10 * actionability, 4)
        confidence = "low" if credibility < settings.feed_min_source_credibility or not info_item.source_url else "high" if final >= 0.6 else "medium"
        return {
            "personal_relevance": relevance,
            "novelty": novelty,
            "cross_domain_distance": cross_domain,
            "opportunity_value": opportunity,
            "source_credibility": credibility,
            "actionability": actionability,
            "final": final,
            "relation_type": relation_type,
            "confidence": confidence,
            "filtered": relevance < settings.feed_min_personal_relevance,
        }


def _profile_terms(profile: Any, field: str) -> list[str]:
    return [str(item).lower() for item in (getattr(profile, field, None) or [])]


def _novelty(published_at: datetime | None) -> float:
    if not published_at:
        return 0.55
    current = datetime.now(UTC).replace(tzinfo=None)
    age = current - published_at
    if age <= timedelta(days=7):
        return 0.90
    if age <= timedelta(days=30):
        return 0.70
    if age <= timedelta(days=90):
        return 0.45
    return 0.25


def _has_opportunity_terms(text: str) -> bool:
    return any(term in text for term in ["agent", "rag", "mcp", "langgraph", "automation", "startup", "benchmark", "open source", "workflow", "productivity", "eval", "research", "github", "paper"])


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))
