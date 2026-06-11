from __future__ import annotations

from collections import Counter
from typing import Any

from src.web_app.rag.bm25 import tokenize
from src.web_app.rag.query_analyzer import QueryAnalysis


WEIGHTS = {
    "exact": {"vector": 0.28, "bm25": 0.58, "keyword": 0.14},
    "table": {"vector": 0.35, "bm25": 0.45, "keyword": 0.20},
    "semantic": {"vector": 0.62, "bm25": 0.28, "keyword": 0.10},
    "summary": {"vector": 0.45, "bm25": 0.35, "keyword": 0.20},
    "document_reference": {"vector": 0.50, "bm25": 0.35, "keyword": 0.15},
    "general": {"vector": 0.55, "bm25": 0.35, "keyword": 0.10},
}


def rerank_results(
    results: list[dict[str, Any]],
    analysis: QueryAnalysis,
    *,
    document_ids: list[int] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not results:
        return []
    parent_counts = Counter(
        (str(item.get("document_id", "")), str(item.get("parent_id") or ""))
        for item in results
        if item.get("parent_id")
    )
    query_tokens = set(tokenize(analysis.query))
    weights = WEIGHTS.get(analysis.query_type, WEIGHTS["general"])
    requested_docs = {str(item) for item in document_ids or []}

    ranked: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        vector_score = _clamp(float(item.get("vector_score") or 0.0))
        bm25_score = _clamp(float(item.get("bm25_score") or 0.0))
        keyword_score = _keyword_score(item, query_tokens)
        final_score = (
            weights["vector"] * vector_score
            + weights["bm25"] * bm25_score
            + weights["keyword"] * keyword_score
        )

        if item.get("retrieval_source") == "hybrid":
            final_score += 0.10
        if analysis.is_exact and item.get("matched_terms"):
            final_score += 0.12
        if analysis.is_table and _is_table_hit(item):
            final_score += 0.12
        if requested_docs and str(item.get("document_id")) in requested_docs:
            final_score += 0.04
        if _filename_or_heading_hit(item, query_tokens):
            final_score += 0.05
        parent_key = (str(item.get("document_id", "")), str(item.get("parent_id") or ""))
        if item.get("parent_id") and parent_counts[parent_key] > 1:
            final_score += min(0.08, 0.025 * parent_counts[parent_key])
        if _is_low_quality_chunk(item):
            final_score -= 0.08

        enriched = dict(item)
        enriched["keyword_score"] = round(keyword_score, 6)
        enriched["final_score"] = round(max(0.0, final_score), 6)
        enriched["_rank_tiebreaker"] = index
        ranked.append(enriched)

    ranked.sort(key=lambda item: (-item["final_score"], item["_rank_tiebreaker"]))
    for item in ranked:
        item.pop("_rank_tiebreaker", None)
    return ranked[:top_k]


def _keyword_score(item: dict[str, Any], query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    haystack = " ".join(
        [
            str(item.get("content", "")),
            str(item.get("filename", "")),
            str(item.get("source_title", "")),
            " ".join(str(part) for part in item.get("heading_path", []) or []),
            " ".join(str(term) for term in item.get("matched_terms", []) or []),
        ]
    )
    tokens = set(tokenize(haystack))
    if not tokens:
        return 0.0
    return len(query_tokens & tokens) / max(1, len(query_tokens))


def _is_table_hit(item: dict[str, Any]) -> bool:
    metadata = item.get("metadata", {}) or {}
    chunk_type = item.get("chunk_type") or metadata.get("chunk_type")
    return bool(chunk_type in {"table", "row_block"} or metadata.get("sheet_name") or item.get("sheet_name") or metadata.get("header"))


def _filename_or_heading_hit(item: dict[str, Any], query_tokens: set[str]) -> bool:
    if not query_tokens:
        return False
    text = " ".join(
        [
            str(item.get("filename", "")),
            str(item.get("source_title", "")),
            " ".join(str(part) for part in item.get("heading_path", []) or []),
        ]
    )
    return bool(query_tokens & set(tokenize(text)))


def _is_low_quality_chunk(item: dict[str, Any]) -> bool:
    content = str(item.get("content") or "")
    if len(content.strip()) < 20:
        return True
    repeated = len(set(content.split())) <= 2 and len(content.split()) > 6
    return repeated


def _clamp(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value
