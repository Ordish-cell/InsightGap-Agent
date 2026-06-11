import hashlib
import logging
from functools import lru_cache
from typing import Any

import requests

from src.web_app.agent.llm.embedding import get_embedding_model
from src.web_app.core.config import settings


MAX_EMBED_CHARS = 8000
PREVIEW_CHARS = 200
DASHSCOPE_MAX_BATCH_SIZE = 10

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"Embedding input must be a string, got {type(text).__name__}")
    cleaned = text.strip()
    if len(cleaned) > MAX_EMBED_CHARS:
        raise ValueError(f"Embedding input is too long: length={len(cleaned)} max={MAX_EMBED_CHARS}")
    return cleaned


def get_embedding_dimension() -> int:
    return settings.qdrant_vector_size or 384


def embed_text(text: str) -> list[float]:
    vectors = embed_texts([text])
    if not vectors:
        raise ValueError("Embedding input is empty after validation")
    return vectors[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned = [_clean_text(text) for text in texts if text is not None and str(text).strip()]
    if not cleaned:
        return []
    model_info = get_embedding_model()
    provider = (model_info.get("provider") or settings.embed_model_type).lower()
    if not model_info.get("model"):
        raise RuntimeError("Embedding model is not configured")
    logger.info("embedding.request provider=%s model=%s input_count=%s length_stats=%s", provider, model_info.get("model"), len(cleaned), _length_stats(cleaned))
    if provider in {"aliyun", "dashscope"}:
        return _embed_dashscope(cleaned)
    if provider in {"sentence-transformers", "sentence_transformers", "local"}:
        return _embed_sentence_transformers(cleaned)
    raise RuntimeError(f"Unsupported embedding provider: {provider}")


def _embed_dashscope(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    api_key = settings.embed_api_key or settings.dashscope_api_key or settings.aliyun_bailian_api_key
    if not api_key:
        raise RuntimeError("DashScope embedding API key is not configured")
    model_info = get_embedding_model()
    model_name = model_info.get("model")
    if not model_name:
        raise RuntimeError("DashScope embedding model is not configured")
    url = (settings.embed_base_url or settings.aliyun_bailian_base_url).rstrip("/") + "/embeddings"
    logger.info(
        "dashscope.embedding.request model=%s input_count=%s length_stats=%s previews=%s",
        model_name,
        len(texts),
        _length_stats(texts),
        _input_previews(texts),
    )
    batch_size = _dashscope_batch_size()
    vectors: list[list[float]] = []
    for batch_index, batch_start in enumerate(range(0, len(texts), batch_size)):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        vectors.extend(_embed_dashscope_batch(url, api_key, model_name, batch_texts, batch_index, batch_start, batch_end))
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding provider returned an unexpected number of vectors: expected={len(texts)} actual={len(vectors)}")
    return vectors


def _dashscope_batch_size() -> int:
    configured = getattr(settings, "dashscope_embedding_batch_size", DASHSCOPE_MAX_BATCH_SIZE) or DASHSCOPE_MAX_BATCH_SIZE
    try:
        batch_size = int(configured)
    except (TypeError, ValueError):
        logger.warning("dashscope.embedding.invalid_batch_size configured=%s default=%s", configured, DASHSCOPE_MAX_BATCH_SIZE)
        return DASHSCOPE_MAX_BATCH_SIZE
    if batch_size <= 0:
        logger.warning("dashscope.embedding.invalid_batch_size configured=%s default=%s", batch_size, DASHSCOPE_MAX_BATCH_SIZE)
        return DASHSCOPE_MAX_BATCH_SIZE
    if batch_size > DASHSCOPE_MAX_BATCH_SIZE:
        logger.warning("dashscope.embedding.batch_size_clamped configured=%s max=%s", batch_size, DASHSCOPE_MAX_BATCH_SIZE)
        return DASHSCOPE_MAX_BATCH_SIZE
    return batch_size


def _embed_dashscope_batch(
    url: str,
    api_key: str,
    model_name: str,
    texts: list[str],
    batch_index: int,
    batch_start: int,
    batch_end: int,
) -> list[list[float]]:
    logger.info(
        "dashscope.embedding.batch_request batch_index=%s batch_start=%s batch_end=%s batch_size=%s model=%s length_stats=%s previews=%s",
        batch_index,
        batch_start,
        batch_end,
        len(texts),
        model_name,
        _length_stats(texts),
        _input_previews(texts),
    )
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model_name, "input": texts},
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        logger.error(
            "dashscope.embedding.http_error batch_index=%s batch_start=%s batch_end=%s batch_size=%s status_code=%s response_text=%s model=%s input_count=%s length_stats=%s previews=%s",
            batch_index,
            batch_start,
            batch_end,
            len(texts),
            response.status_code,
            response.text[:2000],
            model_name,
            len(texts),
            _length_stats(texts),
            _input_previews(texts),
        )
        raise
    logger.info(
        "dashscope.embedding.response_ok batch_index=%s batch_start=%s batch_end=%s batch_size=%s status_code=%s model=%s input_count=%s",
        batch_index,
        batch_start,
        batch_end,
        len(texts),
        response.status_code,
        model_name,
        len(texts),
    )
    payload: dict[str, Any] = response.json()
    vectors = [item["embedding"] for item in sorted(payload.get("data", []), key=lambda item: item.get("index", 0))]
    if len(vectors) != len(texts):
        raise RuntimeError(
            "Embedding provider returned an unexpected number of vectors "
            f"for batch_index={batch_index} batch_start={batch_start} batch_end={batch_end}: "
            f"expected={len(texts)} actual={len(vectors)}"
        )
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


def _length_stats(texts: list[str]) -> dict[str, float | int]:
    lengths = [len(text) for text in texts]
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0, "count": 0}
    return {"min": min(lengths), "max": max(lengths), "avg": round(sum(lengths) / len(lengths), 2), "count": len(lengths)}


def _input_previews(texts: list[str]) -> list[str]:
    previews = []
    for text in texts[:3]:
        compact = " ".join(text.split())
        previews.append(compact[:PREVIEW_CHARS])
    return previews


def deterministic_test_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = [((digest[i % len(digest)] / 255.0) * 2) - 1 for i in range(get_embedding_dimension())]
    return values
