import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_BUCKET_ORDER = ("explicit_related", "adjacent_domain", "far_domain")


def _bucket_of(card: dict) -> str:
    return card.get("relation_type") or card.get("exposure_bucket") or card.get("search_bucket") or "far_domain"


def mix_cards(cards: list[dict], ratio_config: dict | None = None, limit: int = 20) -> tuple[list[dict], dict]:
    """Mix cards by bucket with hard targets. No cross-bucket fallback.

    Hard targets: explicit_related=2, adjacent_domain=2, far_domain=1.
    If ratio_config values > 1, treat as absolute targets.
    Missing buckets are recorded but NOT filled from other buckets.
    """
    ratio = ratio_config or {"explicit_related": 2, "adjacent_domain": 2, "far_domain": 1}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        grouped[_bucket_of(card)].append(card)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.get("final_score", 0), reverse=True)

    bucket_candidate_counts = {key: len(grouped[key]) for key in _BUCKET_ORDER}
    logger.warning("feed mixer candidates: %s", bucket_candidate_counts)

    # Determine targets: absolute if any value > 1, else ratio * limit
    if any(v > 1 for v in ratio.values()):
        targets = {key: int(ratio.get(key, 0)) for key in _BUCKET_ORDER}
    else:
        targets = {key: max(1, int(limit * ratio.get(key, 0))) for key in _BUCKET_ORDER}

    # Select top-N per bucket, interleave buckets to avoid _avoid_repetition dropping entire buckets
    per_bucket: dict[str, list[dict]] = {}
    for key in _BUCKET_ORDER:
        group = grouped[key]
        take = min(targets.get(key, 0), len(group))
        per_bucket[key] = group[:take]

    # Interleave: pick one from each bucket in round-robin so same-domain/same-source cards
    # from the same bucket don't clump together and trigger _avoid_repetition drops.
    mixed: list[dict] = []
    max_per_bucket = max((len(v) for v in per_bucket.values()), default=0)
    for i in range(max_per_bucket):
        for key in _BUCKET_ORDER:
            if i < len(per_bucket[key]):
                mixed.append(per_bucket[key][i])

    mixed = _avoid_repetition(mixed)
    mixed = _limit_low_confidence(mixed, limit)

    # ── Compute bucket_info from ACTUAL final mixed list ──
    actual_selected: dict[str, int] = {"explicit_related": 0, "adjacent_domain": 0, "far_domain": 0}
    for card in mixed:
        b = _bucket_of(card)
        if b in actual_selected:
            actual_selected[b] += 1

    missing_buckets: list[str] = []
    for key in _BUCKET_ORDER:
        short = targets.get(key, 0) - actual_selected.get(key, 0)
        if short > 0:
            missing_buckets.append(key)
            logger.warning("feed mixer: bucket %s short target=%d got=%d (after dedup/confidence filter)", key, targets[key], actual_selected.get(key, 0))

    is_mix_complete = len(missing_buckets) == 0

    # Fill remaining slots from any bucket if below limit
    if len(mixed) < limit:
        remaining = [c for c in cards if c not in mixed]
        remaining.sort(key=lambda item: item.get("final_score", 0), reverse=True)
        for card in remaining:
            if len(mixed) >= limit:
                break
            mixed.append(card)
            b = _bucket_of(card)
            if b in actual_selected:
                actual_selected[b] += 1

    logger.warning("feed mixer final: actual_selected=%s missing=%s is_complete=%s total=%s",
                   actual_selected, missing_buckets or "none", is_mix_complete, len(mixed))
    bucket_info = {
        "candidates": bucket_candidate_counts,
        "selected": actual_selected,
        "missing": missing_buckets,
        "targets": targets,
        "is_complete": is_mix_complete,
    }
    return mixed, bucket_info


def _avoid_repetition(cards: list[dict]) -> list[dict]:
    """Drop cards that would create 3-in-a-row same domain/source, but NEVER drop a card
    from a different bucket than the previous two — bucket diversity trumps repetition avoidance."""
    output: list[dict] = []
    for card in cards:
        if len(output) >= 2:
            prev_two = output[-2:]
            prev_buckets = {_bucket_of(p) for p in prev_two}
            card_bucket = _bucket_of(card)
            # Never drop a card that brings a new bucket — bucket diversity is paramount
            if card_bucket not in prev_buckets:
                output.append(card)
                continue
            # Only avoid repetition within the same bucket
            if all(prev.get("domain") == card.get("domain") for prev in prev_two):
                continue
            if all(prev.get("source_type") == card.get("source_type") for prev in prev_two):
                continue
        output.append(card)
    return output


def _limit_low_confidence(cards: list[dict], limit: int) -> list[dict]:
    """Limit low-confidence cards but never drop the only card for a needed bucket."""
    allowed_low = max(1, int(limit * 0.2))
    low = 0
    output: list[dict] = []
    buckets_seen: set[str] = set()
    for card in cards:
        cb = _bucket_of(card)
        # Never drop the first card of any bucket — preserve diversity
        if card.get("confidence") == "low":
            if low >= allowed_low and cb in buckets_seen:
                continue
            low += 1
        output.append(card)
        buckets_seen.add(cb)
        if len(output) >= limit:
            break
    return output
