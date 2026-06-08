import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_BUCKET_ORDER = ("explicit_related", "adjacent_domain", "far_domain")


def _bucket_of(card: dict) -> str:
    return card.get("relation_type") or card.get("exposure_bucket") or card.get("search_bucket") or "far_domain"


def _get_provider(card: dict) -> str:
    """Extract provider from a card dict."""
    return str(card.get("provider") or card.get("score", {}).get("provider", "")).lower()


def enforce_provider_diversity(cards: list[dict]) -> list[dict]:
    """Enforce provider diversity constraints on the mixed card list.

    Rules:
    - Total github cards <= 2
    - far_domain provider != github
    - adjacent_domain github <= 1
    - Never remove the only card in a bucket — diversity trumps provider constraint.
    """
    # Count per-bucket github cards
    bucket_cards: dict[str, list[int]] = {"explicit_related": [], "adjacent_domain": [], "far_domain": []}
    github_total = 0
    for i, card in enumerate(cards):
        b = _bucket_of(card)
        if b not in bucket_cards:
            continue
        bucket_cards[b].append(i)
        if _get_provider(card) == "github":
            github_total += 1

    if github_total <= 2:
        # Still check far_domain must not be github
        to_drop: set[int] = set()
        for b in ("far_domain",):
            bucket_indices = [idx for idx in bucket_cards.get(b, [])]
            github_in_bucket = [idx for idx in bucket_indices if _get_provider(cards[idx]) == "github"]
            non_github_in_bucket = [idx for idx in bucket_indices if _get_provider(cards[idx]) != "github"]
            for idx in github_in_bucket:
                if len(non_github_in_bucket) + len(github_in_bucket) - len([x for x in github_in_bucket if x in to_drop or x == idx]) > 0:
                    # There's at least one non-github card in this bucket, safe to drop github
                    to_drop.add(idx)
                    logger.warning("feed mixer provider diversity: dropping far_domain github card title=%s", (cards[idx].get("title") or "")[:80])
                else:
                    logger.warning("feed mixer provider diversity: keeping far_domain github card (only card in bucket) title=%s", (cards[idx].get("title") or "")[:80])
        if to_drop:
            cards = [c for i, c in enumerate(cards) if i not in to_drop]
        return cards

    # More than 2 github — need to drop excess
    to_drop: set[int] = set()
    # Priority for dropping: far_domain github > adjacent_domain github > explicit_related github
    drop_order: list[tuple[int, int, str]] = []  # (priority, index, bucket)
    bucket_priority = {"far_domain": 0, "adjacent_domain": 1, "explicit_related": 2}
    for i, card in enumerate(cards):
        if _get_provider(card) == "github":
            b = _bucket_of(card)
            drop_order.append((bucket_priority.get(b, 9), i, b))

    drop_order.sort(key=lambda x: x[0])  # lowest priority first = drop far then adjacent then explicit

    # Ensure each bucket keeps at least 1 card after drops
    bucket_remaining = {b: len([idx for idx in bucket_cards[b] if idx not in to_drop]) for b in bucket_cards}

    for _, idx, bkt in drop_order:
        if github_total - len(to_drop) <= 2:
            break
        # Don't drop if it would empty the bucket
        if bucket_remaining.get(bkt, 0) <= 1:
            logger.warning("feed mixer provider diversity: keeping github card (last in bucket=%s) title=%s", bkt, (cards[idx].get("title") or "")[:80])
            continue
        to_drop.add(idx)
        bucket_remaining[bkt] -= 1
        logger.warning("feed mixer provider diversity: dropping github card bucket=%s title=%s", bkt, (cards[idx].get("title") or "")[:80])

    # Also enforce far_domain != github
    far_indices = bucket_cards.get("far_domain", [])
    for idx in far_indices:
        if idx in to_drop:
            continue
        if _get_provider(cards[idx]) == "github":
            non_github_far = [j for j in far_indices if j not in to_drop and _get_provider(cards[j]) != "github"]
            if non_github_far:
                to_drop.add(idx)
                logger.warning("feed mixer provider diversity: dropping far_domain github card title=%s", (cards[idx].get("title") or "")[:80])
            else:
                logger.warning("feed mixer provider diversity: cannot drop far_domain github (no non-github far) title=%s", (cards[idx].get("title") or "")[:80])

    # Enforce adjacent_domain github <= 1
    adj_indices = bucket_cards.get("adjacent_domain", [])
    adj_github = [idx for idx in adj_indices if idx not in to_drop and _get_provider(cards[idx]) == "github"]
    while len(adj_github) > 1:
        idx = adj_github.pop()
        if bucket_remaining.get("adjacent_domain", 0) <= 1:
            break
        to_drop.add(idx)
        bucket_remaining["adjacent_domain"] -= 1
        logger.warning("feed mixer provider diversity: dropping adjacent_domain github card title=%s", (cards[idx].get("title") or "")[:80])

    if to_drop:
        cards = [c for i, c in enumerate(cards) if i not in to_drop]
        logger.warning("feed mixer provider diversity: dropped %d github cards, remaining=%d", len(to_drop), len(cards))

    return cards


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

    # ── Enforce provider diversity BEFORE interleaving ──
    # Flatten per_bucket into a list, enforce, then re-split
    all_selected: list[dict] = []
    for key in _BUCKET_ORDER:
        all_selected.extend(per_bucket[key])
    all_selected = enforce_provider_diversity(all_selected)

    # Re-split after diversity enforcement
    per_bucket = {"explicit_related": [], "adjacent_domain": [], "far_domain": []}
    for card in all_selected:
        b = _bucket_of(card)
        if b in per_bucket:
            per_bucket[b].append(card)

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
