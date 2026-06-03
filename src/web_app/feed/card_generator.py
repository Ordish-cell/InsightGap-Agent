from typing import Any


def generate_feed_card(info_item: Any, score: dict[str, Any], user_profile: Any) -> dict[str, Any]:
    domain = (info_item.raw_metadata or {}).get("domain", "ai")
    tags = (info_item.raw_metadata or {}).get("tags", info_item.topics or [])
    interests = getattr(user_profile, "explicit_interests", None) or ["Agent", "RAG"]
    matched = ", ".join([tag for tag in tags if str(tag).lower() in " ".join(interests).lower()]) or ", ".join(interests[:2])
    evidence = [
        {
            "title": info_item.title,
            "url": info_item.source_url or None,
            "source_type": info_item.source_type,
            "credibility": score["source_credibility"],
            "published_at": info_item.published_at.isoformat() if info_item.published_at else None,
            "snippet": info_item.summary[:300],
        }
    ]
    return {
        "card_type": "insight",
        "title": info_item.title,
        "one_sentence_value": f"这条信息可能影响你在 {domain} 方向的判断，因为它涉及 {', '.join(tags[:3]) or domain} 的新变化。",
        "why_you": f"你当前关注 {', '.join(interests[:4])}，这条信息与 {matched} 有交集。",
        "information_gap": f"多数人可能只把它当作普通更新，但它背后可能暗示 {domain} 方向的新机会或风险。",
        "summary": info_item.summary,
        "source_type": info_item.source_type,
        "domain": domain,
        "relation_type": score["relation_type"],
        "evidence": evidence,
        "suggested_actions": ["save", "ignore", "deep_research", "generate_report", "create_skill_draft"],
        "score": score,
        "final_score": score["final"],
        "confidence": score["confidence"],
        "status": "active",
    }
