from collections import defaultdict


def mix_cards(cards: list[dict], ratio_config: dict | None = None, limit: int = 20) -> list[dict]:
    ratio = ratio_config or {"explicit_related": 0.30, "adjacent_domain": 0.40, "far_domain": 0.30}
    grouped = defaultdict(list)
    for card in cards:
        grouped[card.get("relation_type", "far_domain")].append(card)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.get("final_score", 0), reverse=True)

    targets = {key: max(1, int(limit * value)) for key, value in ratio.items()}
    mixed: list[dict] = []
    for key in ("explicit_related", "adjacent_domain", "far_domain"):
        mixed.extend(grouped[key][: targets.get(key, 0)])

    remaining = [card for rows in grouped.values() for card in rows if card not in mixed]
    remaining.sort(key=lambda item: item.get("final_score", 0), reverse=True)
    mixed.extend(remaining)
    mixed = _avoid_repetition(mixed)
    return _limit_low_confidence(mixed, limit)


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
