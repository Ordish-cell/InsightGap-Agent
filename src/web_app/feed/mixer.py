import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def mix_cards(cards: list[dict], ratio_config: dict | None = None, limit: int = 20) -> tuple[list[dict], dict]:
    """Mix cards by bucket ratio. Returns (mixed_cards, bucket_info)."""
    ratio = ratio_config or {"explicit_related": 0.30, "adjacent_domain": 0.40, "far_domain": 0.30}
    grouped = defaultdict(list)
    for card in cards:
        grouped[card.get("relation_type", "far_domain")].append(card)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.get("final_score", 0), reverse=True)

    bucket_candidate_counts = {key: len(grouped[key]) for key in ("explicit_related", "adjacent_domain", "far_domain")}
    logger.info("feed mixer candidates: %s", bucket_candidate_counts)

    # If values are > 1, treat as absolute targets; otherwise as ratio
    if any(v > 1 for v in ratio.values()):
        targets = {key: value for key, value in ratio.items()}
    else:
        targets = {key: max(1, int(limit * value)) for key, value in ratio.items()}
    mixed: list[dict] = []
    missing_buckets: list[str] = []
    bucket_selected: dict[str, int] = {}

    for key in ("explicit_related", "adjacent_domain", "far_domain"):
        group = grouped[key]
        take = min(targets.get(key, 0), len(group))
        if take == 0 and targets.get(key, 0) > 0:
            missing_buckets.append(key)
            logger.warning("feed mixer: bucket %s has 0 candidates (target %d)", key, targets[key])
        mixed.extend(group[:take])
        bucket_selected[key] = take

    remaining = [card for rows in grouped.values() for card in rows if card not in mixed]
    remaining.sort(key=lambda item: item.get("final_score", 0), reverse=True)
    mixed.extend(remaining)
    mixed = _avoid_repetition(mixed)
    mixed = _limit_low_confidence(mixed, limit)

    logger.info("feed mixer final: selected=%s missing=%s", bucket_selected, missing_buckets or "none")
    bucket_info = {"candidates": bucket_candidate_counts, "selected": bucket_selected, "missing": missing_buckets, "targets": targets}
    return mixed, bucket_info


def _avoid_repetition(cards: list[dict]) -> list[dict]:
    output: list[dict] = []
    for card in cards:
        if len(output) >= 2 and all(prev.get("domain") == card.get("domain") for prev in output[-2:]):
            continue
        if len(output) >= 2 and all(prev.get("source_type") == card.get("source_type") for prev in output[-2:]):
            continue
        output.append(card)
    return output


def _limit_low_confidence(cards: list[dict], limit: int) -> list[dict]:
    allowed_low = max(1, int(limit * 0.2))
    low = 0
    output = []
    for card in cards:
        if card.get("confidence") == "low":
            if low >= allowed_low:
                continue
            low += 1
        output.append(card)
        if len(output) >= limit:
            break
    return output
