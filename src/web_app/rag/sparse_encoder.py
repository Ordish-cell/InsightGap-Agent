from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from src.web_app.core.config import settings
from src.web_app.rag.bm25 import tokenize


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]

    def is_empty(self) -> bool:
        return not self.indices or not self.values


class HashingSparseEncoder:
    """Dependency-free sparse encoder for Qdrant sparse/BM25-style retrieval.

    Qdrant applies collection-level IDF when SparseVectorParams.modifier=IDF.
    This encoder supplies stable hashed token term weights.
    """

    def __init__(self, hash_size: int | None = None):
        self.hash_size = int(hash_size or settings.qdrant_sparse_hash_size or 2_000_003)

    def encode(self, text: str) -> SparseVectorData:
        tokens = tokenize(text or "")
        if not tokens:
            return SparseVectorData(indices=[], values=[])
        counts = Counter(tokens)
        weighted: dict[int, float] = {}
        for token, count in counts.items():
            index = self._index(token)
            value = 1.0 + math.log(float(count))
            weighted[index] = weighted.get(index, 0.0) + value
        indices = sorted(weighted)
        values = [weighted[index] for index in indices]
        return SparseVectorData(indices=indices, values=values)

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.hash_size


def get_sparse_encoder() -> HashingSparseEncoder:
    return HashingSparseEncoder()


def build_sparse_document_input(text: str) -> Any:
    """Build the Qdrant sparse vector input for document upsert.

    The cloud BM25 path returns a Qdrant inference input, not a precomputed
    vector. The hashing path remains available for legacy/eval collections.
    """
    return _build_sparse_input(text)


def build_sparse_query_input(text: str) -> Any:
    """Build the Qdrant sparse vector input for hybrid query prefetch."""
    return _build_sparse_input(text)


def is_sparse_input_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, models.Document):
        return not bool((value.text or "").strip())
    if isinstance(value, models.SparseVector):
        return not bool(value.indices and value.values)
    return False


def _build_sparse_input(text: str) -> Any:
    encoder = (settings.qdrant_sparse_encoder or "hashing_sparse").lower()
    if encoder == "qdrant_cloud_bm25":
        model_name = settings.qdrant_sparse_model or "Qdrant/bm25"
        return models.Document(text=text or "", model=model_name)
    if encoder in {"hashing_sparse", "hashing", "legacy_hashing"}:
        sparse = get_sparse_encoder().encode(text or "")
        return models.SparseVector(indices=sparse.indices, values=sparse.values)
    raise RuntimeError(f"Unsupported Qdrant sparse encoder: {settings.qdrant_sparse_encoder}")
