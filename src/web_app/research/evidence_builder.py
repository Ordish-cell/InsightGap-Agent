from typing import Any


def evidence_from_feed_card(card: Any) -> list[dict[str, Any]]:
    rows = []
    for item in card.evidence or []:
        rows.append(
            {
                "source_type": item.get("source_type", "feed_card"),
                "title": item.get("title") or card.title,
                "url": item.get("url"),
                "snippet": item.get("snippet") or card.one_sentence_value,
                "score": item.get("credibility", 0.7),
                "metadata": {"feed_card_id": card.id, "relation_type": card.exposure_bucket},
            }
        )
    if not rows:
        rows.append({"source_type": "feed_card", "title": card.title, "url": None, "snippet": card.one_sentence_value, "score": 0.4, "metadata": {"feed_card_id": card.id}})
    return rows


def evidence_from_rag_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_type": "rag",
            "title": item.get("source_title", "RAG chunk"),
            "url": item.get("source_url"),
            "snippet": item.get("content_preview") or item.get("content", "")[:500],
            "score": item.get("score", 0.0),
            "metadata": {"document_id": item.get("document_id"), "chunk_id": item.get("chunk_id"), **(item.get("metadata") or {})},
        }
        for item in results
    ]
