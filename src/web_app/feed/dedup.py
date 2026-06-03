from src.web_app.feed.normalizer import InfoItemCreate


def deduplicate_items(items: list[InfoItemCreate]) -> tuple[list[InfoItemCreate], int]:
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    unique: list[InfoItemCreate] = []
    skipped = 0
    for item in items:
        canonical_url = item.raw_metadata.get("canonical_url") or ""
        key = item.content_hash
        if key in seen_hashes or (canonical_url and canonical_url in seen_urls):
            skipped += 1
            continue
        seen_hashes.add(key)
        if canonical_url:
            seen_urls.add(canonical_url)
        unique.append(item)
    return unique, skipped
