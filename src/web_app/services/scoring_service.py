from typing import Any


WEIGHTS = {
    "personal_relevance": 0.30,
    "novelty": 0.20,
    "cross_domain_distance": 0.15,
    "opportunity_value": 0.15,
    "source_credibility": 0.10,
    "actionability": 0.10,
}


def calculate_final_score(score_detail: dict[str, float]) -> float:
    return round(sum(score_detail.get(name, 0.0) * weight for name, weight in WEIGHTS.items()), 4)


def classify_exposure_bucket(info_item: Any, user_profile: Any) -> str:
    topics = set(getattr(info_item, "topics", []) or info_item.get("topics", []))
    explicit = set(getattr(user_profile, "explicit_interests", []) or user_profile.get("explicit_interests", []))
    adjacent = set(getattr(user_profile, "adjacent_domains", []) or user_profile.get("adjacent_domains", []))
    if topics & explicit:
        return "explicit_related"
    if topics & adjacent:
        return "adjacent_domain"
    return "far_domain"


def score_info_item_for_user(info_item: Any, user_profile: Any) -> dict[str, float | str]:
    topics = set(getattr(info_item, "topics", []) or info_item.get("topics", []))
    source_url = getattr(info_item, "source_url", "") or info_item.get("source_url", "")
    disliked = set(getattr(user_profile, "disliked_topics", []) or user_profile.get("disliked_topics", []))
    explicit = set(getattr(user_profile, "explicit_interests", []) or user_profile.get("explicit_interests", []))
    adjacent = set(getattr(user_profile, "adjacent_domains", []) or user_profile.get("adjacent_domains", []))
    far = set(getattr(user_profile, "far_domains", []) or user_profile.get("far_domains", []))

    personal_relevance = 0.25 + (0.45 if topics & explicit else 0.0) + (0.20 if topics & adjacent else 0.0)
    if topics & disliked:
        personal_relevance *= 0.4
    score_detail = {
        "personal_relevance": min(personal_relevance, 1.0),
        "novelty": 0.70,
        "cross_domain_distance": 0.75 if topics & far else 0.45,
        "opportunity_value": 0.60,
        "source_credibility": 0.80 if source_url else 0.35,
        "actionability": 0.65,
    }
    score_detail["final"] = calculate_final_score(score_detail)
    if score_detail["source_credibility"] < 0.4:
        score_detail["verification_status"] = "unverified"
    return score_detail


def should_show_card(score_detail: dict[str, float]) -> bool:
    return score_detail.get("personal_relevance", 0.0) >= 0.15


def deduplicate_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for card in cards:
        key = (card.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(card)
    return unique
