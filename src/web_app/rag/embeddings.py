import hashlib
from functools import lru_cache
from typing import Any

import requests

from src.web_app.agent.llm.embedding import get_embedding_model
from src.web_app.core.config import settings


MAX_EMBED_CHARS = 8000


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Embedding input must be a string")
    return text.strip()[:MAX_EMBED_CHARS]


def get_embedding_dimension() -> int:
    return settings.qdrant_vector_size or 384


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned = [_clean_text(text) for text in texts]
    if not cleaned:
        return []
    if any(not text for text in cleaned):
        return [_zero_vector() for _ in cleaned]
    model_info = get_embedding_model()
    provider = (model_info.get("provider") or settings.embed_model_type).lower()
    if provider in {"aliyun", "dashscope"}:
        return _embed_dashscope(cleaned)
    if provider in {"sentence-transformers", "sentence_transformers", "local"}:
        return _embed_sentence_transformers(cleaned)
    raise RuntimeError(f"Unsupported embedding provider: {provider}")


def _embed_dashscope(texts: list[str]) -> list[list[float]]:
    api_key = settings.embed_api_key or settings.dashscope_api_key or settings.aliyun_bailian_api_key
    if not api_key:
        raise RuntimeError("DashScope embedding API key is not configured")
    model_info = get_embedding_model()
    url = (settings.embed_base_url or settings.aliyun_bailian_base_url).rstrip("/") + "/embeddings"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model_info["model"], "input": texts},
        timeout=30,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    vectors = [item["embedding"] for item in sorted(payload.get("data", []), key=lambda item: item.get("index", 0))]
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding provider returned an unexpected number of vectors")
    return [_fit_dimension(vector) for vector in vectors]


@lru_cache
def _sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.agent_embedding_model or settings.embed_model_name or "sentence-transformers/all-MiniLM-L6-v2")


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    vectors = _sentence_model().encode(texts, normalize_embeddings=True)
    return [_fit_dimension([float(value) for value in vector]) for vector in vectors]


def _fit_dimension(vector: list[float]) -> list[float]:
    dimension = get_embedding_dimension()
    if len(vector) == dimension:
        return vector
    if len(vector) > dimension:
        return vector[:dimension]
    return vector + [0.0] * (dimension - len(vector))


def _zero_vector() -> list[float]:
    return [0.0] * get_embedding_dimension()


def deterministic_test_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = [((digest[i % len(digest)] / 255.0) * 2) - 1 for i in range(get_embedding_dimension())]
    return values
