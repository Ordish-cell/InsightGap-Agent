from datetime import UTC, datetime, timedelta
from typing import Any

from src.web_app.core.config import settings


class FeedScorer:
    def score(
        self,
        info_item: Any,
        user_profile: Any,
        user_feedback_stats: dict | None = None,
        semantic_memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        feedback = user_feedback_stats or {}
        topics = [str(item).lower() for item in (info_item.topics or [])]
        text = f"{info_item.title} {info_item.summary} {' '.join(topics)}".lower()
        explicit = _profile_terms(user_profile, "explicit_interests") + _profile_terms(user_profile, "goals")
        adjacent = _profile_terms(user_profile, "adjacent_domains")
        far = _profile_terms(user_profile, "far_domains")

        profile_match = _compute_profile_match(text, explicit, adjacent, far)
        semantic_memory_match = _compute_memory_match(text, semantic_memories or [])
        recent_feedback_match = _compute_feedback_match(text, topics, feedback)
        skill_interest_match = _compute_skill_interest_match(text, topics)
        negative_penalty = _compute_negative_penalty(text, topics, user_profile, semantic_memories or [])

        personal_relevance = (
            0.35 * profile_match
            + 0.35 * semantic_memory_match
            + 0.15 * recent_feedback_match
            + 0.10 * skill_interest_match
            + 0.05 * negative_penalty
        )
        personal_relevance = _clamp(personal_relevance)

        if profile_match >= 0.55:
            relation_type = "explicit_related"
        elif profile_match >= 0.30 or semantic_memory_match >= 0.30:
            relation_type = "adjacent_domain"
        else:
            relation_type = "far_domain"

        credibility = float((info_item.raw_metadata or {}).get("source_credibility", 0.40))
        novelty = _novelty(info_item.published_at)
        cross_domain = {"explicit_related": 0.35, "adjacent_domain": 0.70, "far_domain": 0.82}[relation_type]
        opportunity = 0.75 if _has_opportunity_terms(text) else 0.45
        actionability = 0.75 if any(term in text for term in ["github", "tool", "framework", "workflow", "agent", "rag"]) else 0.50
        final = round(
            0.30 * personal_relevance + 0.20 * novelty + 0.15 * cross_domain + 0.15 * opportunity + 0.10 * credibility + 0.10 * actionability,
            4,
        )
        confidence = "low" if credibility < settings.feed_min_source_credibility or not info_item.source_url else "high" if final >= 0.6 else "medium"
        return {
            "personal_relevance": personal_relevance,
            "novelty": novelty,
            "cross_domain_distance": cross_domain,
            "opportunity_value": opportunity,
            "source_credibility": credibility,
            "actionability": actionability,
            "final": final,
            "relation_type": relation_type,
            "confidence": confidence,
            "filtered": personal_relevance < settings.feed_min_personal_relevance,
            "profile_match": profile_match,
            "semantic_memory_match": semantic_memory_match,
        }


def _compute_profile_match(text: str, explicit: list[str], adjacent: list[str], far: list[str]) -> float:
    explicit_hits = [term for term in explicit if term and term in text]
    adjacent_hits = [term for term in adjacent if term and term in text]
    far_hits = [term for term in far if term and term in text]
    if explicit_hits:
        return 0.65 + 0.05 * min(len(explicit_hits), 3)
    if adjacent_hits:
        return 0.45 + 0.05 * min(len(adjacent_hits), 3)
    if far_hits:
        return 0.30 + 0.03 * min(len(far_hits), 3)
    return 0.22 if _has_opportunity_terms(text) else 0.12


def _compute_memory_match(text: str, memories: list[dict[str, Any]]) -> float:
    if not memories:
        return 0.15
    hits = 0
    total = 0
    for mem in memories:
        if not mem or not mem.get("content"):
            continue
        total += 1
        content = str(mem["content"]).lower()
        memory_terms = _extract_significant_terms(content)
        for term in memory_terms:
            if term and term in text:
                hits += 1
                break
    if total == 0:
        return 0.15
    ratio = hits / total
    return _clamp(0.10 + 0.55 * ratio)


def _compute_feedback_match(text: str, topics: list[str], feedback: dict) -> float:
    positive = feedback.get("positive_topics", {}) or {}
    negative = feedback.get("negative_topics", {}) or {}
    score = 0.30
    for topic in topics:
        score += 0.08 * positive.get(topic, 0)
        score -= 0.10 * negative.get(topic, 0)
    return _clamp(score)


def _compute_skill_interest_match(text: str, topics: list[str]) -> float:
    skill_terms = ["skill", "workflow", "reusable", "tool", "mcp", "automation", "template", "recipe", "pipeline"]
    hits = sum(1 for term in skill_terms if term in text)
    topic_hits = sum(1 for t in topics if t.lower() in skill_terms)
    return _clamp(0.20 + 0.15 * hits + 0.10 * topic_hits)


def _compute_negative_penalty(text: str, topics: list[str], profile: Any, memories: list[dict[str, Any]]) -> float:
    disliked = [str(item).lower() for item in (getattr(profile, "disliked_topics", None) or [])]
    penalty = 0.60
    for term in disliked:
        if term and term in text:
            penalty -= 0.20
    for mem in memories:
        content = str(mem.get("content", "")).lower()
        if "不喜欢" in content or "不要" in content or "避免" in content:
            for term in _extract_significant_terms(content):
                if term and term in text:
                    penalty -= 0.15
                    break
    exa_terms = ["exa", "neo4j", "neo4j"]
    if any(t in text for t in exa_terms):
        penalty += 0.10
    return _clamp(penalty)


def _extract_significant_terms(content: str) -> list[str]:
    import re
    words = re.findall(r"[a-z一-鿿]{2,}", content.lower())
    stop = {"the", "and", "for", "with", "from", "that", "this", "have", "been", "were", "are", "not", "but", "its", "was", "has", "had", "can", "all", "will", "just", "also", "into", "more", "some", "such", "than", "then", "them", "over", "only", "like", "when", "make", "made", "most", "much", "very", "your", "about", "after", "their", "there", "which", "would", "could", "other", "being", "doing"}
    return [w for w in words if w not in stop][:8]


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
