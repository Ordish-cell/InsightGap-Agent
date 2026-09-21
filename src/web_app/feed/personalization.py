"""Deterministic, source-attributed relevance without inferred user identities."""

import re
import unicodedata
from typing import Any

NO_PERSONALIZATION = "目前没有足够的用户资料或已保存记忆，能够说明这条信息与你的具体关联。"
PROFILE_FIELDS = ("explicit_interests", "goals", "adjacent_domains", "far_domains")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "i", "in", "is", "it", "my", "of", "on", "or", "the", "this", "to", "use",
    "uses", "using", "was", "with", "you", "your", "interested", "interests",
    "want", "work", "working", "project", "prefer", "like",
    "使用", "关注", "用户", "喜欢", "研究", "项目", "目标", "计划", "希望",
}
_NEGATIVE = re.compile(
    r"不喜欢|不感兴趣|不关注|不使用|不用|不做|不考虑|不参与|不再|不要|避免|讨厌|拒绝|并非|不是|未使用|没有使用"
    r"|\b(?:not|no|never|avoid|dislike|dislikes|without)\b|\b\w+n['’]t\b", re.I
)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _contains(text: str, term: str) -> bool:
    # ASCII boundaries permit Chinese text adjacent to a Latin technology name.
    left = r"(?<![a-z0-9_])" if term[0].isascii() else ""
    right = r"(?![a-z0-9_])" if term[-1].isascii() else ""
    return bool(re.search(left + re.escape(term) + right, text))


def _terms(text: str) -> set[str]:
    # Keep Chinese phrases whole; do not manufacture overlapping two-character words.
    return {
        t for t in re.findall(r"[a-z][a-z0-9]*(?:[.+#-][a-z0-9]+)*|[\u4e00-\u9fff]{2,}", text)
        if len(t) >= 2 and t not in _STOP_WORDS
    }


def _matches(fact: str, signal: str, tags: list[str]) -> list[str]:
    fact = _normalize(fact)
    # Explicit signal tags also provide complete Chinese terms within longer facts.
    candidates = _terms(fact) | {
        _normalize(t) for t in tags
        if isinstance(t, str) and _terms(_normalize(t))
    }
    return sorted(t for t in candidates if t not in _STOP_WORDS and _contains(fact, t) and _contains(signal, t))


def match_user_facts(info_item: Any, profile: Any, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tags = list((info_item.raw_metadata or {}).get("tags") or []) + list(info_item.topics or [])
    signal = _normalize(" ".join([info_item.title or "", info_item.summary or "", *tags]))
    candidates = []
    for field in PROFILE_FIELDS:
        for value in getattr(profile, field, None) or []:
            if isinstance(value, str) and value.strip():
                candidates.append({"source_type": "profile", "profile_field": field, "text": value.strip()})

    user_id = getattr(profile, "user_id", None)
    for memory in sorted(memories, key=lambda m: str(m.get("id", ""))):
        meta = memory.get("metadata") or {}
        if (user_id is None or memory.get("user_id") != user_id
                or memory.get("memory_type") != "semantic" or memory.get("id") is None
                or meta.get("status", "active") != "active" or meta.get("sensitive")
                or memory.get("_low_confidence") or meta.get("category") == "negative_preference"):
            continue
        try:
            confidence = float(meta.get("confidence", 1.0))
        except (TypeError, ValueError):
            continue
        if not 0.55 <= confidence <= 1.0:
            continue
        value = memory.get("content")
        if isinstance(value, str) and value.strip():
            candidates.append({"source_type": "memory", "memory_id": memory["id"], "text": value.strip()})

    evidence, seen = [], set()
    disliked = [_normalize(t) for t in getattr(profile, "disliked_topics", None) or [] if isinstance(t, str) and t.strip()]
    for fact in candidates:
        text = _normalize(fact["text"])
        if text in seen or _NEGATIVE.search(text) or any(_contains(text, d) for d in disliked):
            continue
        matched = _matches(text, signal, tags)
        if not matched:
            continue
        seen.add(text)
        evidence.append({**fact, "matched_terms": matched})
        if len(evidence) == 2:
            break
    return evidence


def render_relevance(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return NO_PERSONALIZATION
    labels = {"explicit_interests": "关注", "goals": "目标", "adjacent_domains": "邻近领域", "far_domains": "远域关注"}
    parts = []
    for fact in evidence[:2]:
        prefix = f"画像中的{labels[fact['profile_field']]}记录为" if fact["source_type"] == "profile" else "已保存记忆提到"
        parts.append(f"{prefix}『{fact['text']}』，该信号也涉及 {'、'.join(fact['matched_terms'])}。")
    return "".join(parts)
