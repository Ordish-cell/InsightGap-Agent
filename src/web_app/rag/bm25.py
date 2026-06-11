from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


TOKEN_RE = re.compile(
    r"https?://[^\s)）]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|"
    r"\b[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)+\b|"
    r"\b[A-Za-z_][A-Za-z0-9_]*\b|"
    r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?\b|"
    r"\b1[3-9]\d{9}\b|"
    r"\b\d+(?:\.\d+)?\b|"
    r"[\u4e00-\u9fff]+"
)


@dataclass
class BM25Document:
    id: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25Hit:
    document: BM25Document
    score: float
    normalized_score: float
    matched_terms: list[str]


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        raw = match.group(0).strip()
        if not raw:
            continue
        if _is_cjk(raw):
            tokens.extend(_cjk_ngrams(raw))
        else:
            tokens.append(raw.lower())
    if not tokens and text.strip():
        tokens.append(text.strip().lower())
    return _dedupe_keep_repeats(tokens)


def bm25_search(
    query: str,
    documents: Iterable[BM25Document],
    *,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[BM25Hit]:
    docs = list(documents)
    if not query or not docs or top_k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    doc_tokens = [tokenize(doc.content) for doc in docs]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    if avg_len <= 0:
        return []

    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    scores: list[tuple[BM25Document, float, list[str]]] = []
    n_docs = len(docs)
    for doc, tokens, doc_len in zip(docs, doc_tokens, doc_lengths, strict=True):
        if not tokens:
            continue
        tf = Counter(tokens)
        score = 0.0
        matched: list[str] = []
        for term in query_tokens:
            term_tf = tf.get(term, 0)
            if term_tf <= 0:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = term_tf + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (term_tf * (k1 + 1)) / denom
            matched.append(term)
        if score > 0:
            scores.append((doc, score, _unique(matched)))

    if not scores:
        return []
    max_score = max(score for _, score, _ in scores) or 1.0
    ranked = sorted(scores, key=lambda item: (-item[1], item[0].id))
    return [
        BM25Hit(document=doc, score=score, normalized_score=score / max_score, matched_terms=matched)
        for doc, score, matched in ranked[:top_k]
    ]


def _cjk_ngrams(text: str) -> list[str]:
    chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    if not chars:
        return []
    if len(chars) == 1:
        return chars
    grams: list[str] = []
    grams.extend("".join(chars[i:i + 2]) for i in range(len(chars) - 1))
    if len(chars) >= 3:
        grams.extend("".join(chars[i:i + 3]) for i in range(len(chars) - 2))
    grams.extend(chars)
    return grams


def _is_cjk(text: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in text)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _dedupe_keep_repeats(tokens: list[str]) -> list[str]:
    # Preserve repeated terms for term frequency, while removing empty artifacts.
    return [token for token in tokens if token]
