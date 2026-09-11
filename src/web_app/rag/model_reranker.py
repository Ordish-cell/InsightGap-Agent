"""Bounded, cancellable Jina reranking with atomic fallback.

The synchronous retrieval service bridges to one dedicated HTTP event loop.
There is no retry or waiting queue: at most four requests per application process.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar
import math
import hashlib
import json
from threading import BoundedSemaphore, Lock, Thread
import ssl
from time import perf_counter
from typing import Any

import httpx

from src.web_app.core.config import settings
from src.web_app.rag.bm25 import tokenize

_slots = BoundedSemaphore(4)
_loop = None
_loop_lock = Lock()
_client = None
_initialization = None
_mode = ContextVar("rag_rerank_mode", default=None)


@contextmanager
def rerank_mode(mode):
    """Request-local override for isolated evaluations; never changes Settings."""
    token = _mode.set(mode)
    try:
        yield
    finally:
        _mode.reset(token)


def _enabled(user_id=None):
    if (_mode.get() or settings.rag_rerank_mode) != "model":
        return False
    allowed = {s.strip() for s in settings.rag_rerank_user_ids.split(",") if s.strip()}
    return user_id is None or not allowed or str(user_id) in allowed


async def _initialize_client():
    global _client
    # SSL trust-store loading is blocking and expensive on Windows. Build it
    # once, outside the request deadline and outside either HTTP event loop.
    def create():
        import os
        import certifi
        context = ssl.create_default_context(
            cafile=os.environ.get("SSL_CERT_FILE") or (None if os.environ.get("SSL_CERT_DIR") else certifi.where()),
            capath=os.environ.get("SSL_CERT_DIR"))
        return httpx.AsyncClient(verify=context,
            timeout=httpx.Timeout(settings.rag_rerank_timeout_ms / 1000,
                                  connect=min(settings.rag_rerank_connect_timeout_ms, settings.rag_rerank_timeout_ms) / 1000),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4,
                               keepalive_expiry=settings.rag_rerank_keepalive_seconds),
            follow_redirects=False)
    _client = await asyncio.to_thread(create)


def prepare_reranker_transport(user_id=None):
    """Start local client initialization; no network request or model charge.

    Called before embedding (and at application startup). Callers never wait
    on initialization inside the 900ms rerank budget.
    """
    global _loop, _initialization
    if not _enabled(user_id) or not settings.rag_rerank_api_key or not settings.rag_rerank_endpoint:
        return None
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            Thread(target=_loop.run_forever, name="rag-rerank-http", daemon=True).start()
        if _initialization is None or (_initialization.done() and _initialization.exception() is not None):
            _initialization = asyncio.run_coroutine_threadsafe(_initialize_client(), _loop)
        return _initialization


def query_window(text: str, query: str, byte_budget: int) -> tuple[str, bool]:
    raw = text.encode("utf-8")
    if len(raw) <= byte_budget:
        return text, False
    # Prefer a concrete query term; retain a window, not only the chunk prefix.
    terms = sorted(set(tokenize(query)), key=len, reverse=True)
    lower = text.casefold()
    position = next((lower.find(t.casefold()) for t in terms if len(t) > 1 and t.casefold() in lower), 0)
    byte_position = len(text[:position].encode("utf-8"))
    start = max(0, min(byte_position - byte_budget // 4, len(raw) - byte_budget))
    return raw[start:start + byte_budget].decode("utf-8", errors="ignore"), True


def rerank_input(hit, query, byte_budget=1000):
    metadata = hit.get("metadata") or {}
    header = "\n".join(str(x) for x in [hit.get("filename") or hit.get("source_title", ""),
        " > ".join(hit.get("heading_path") or metadata.get("heading_path") or []),
        metadata.get("version", ""), metadata.get("effective_date", "")] if x)
    header = header.encode("utf-8")[:200].decode("utf-8", errors="ignore")
    content, truncated = query_window(str(hit.get("content") or hit.get("content_preview") or ""),
                                      query, max(1, byte_budget - len(header.encode("utf-8")) - 1))
    return (header + "\n" + content).strip(), truncated


def validate_results(payload: Any, count: int):
    rows = payload["results"]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("incomplete_results")
    seen = set()
    scores = {}
    for row in rows:
        index, score = row["index"], row["relevance_score"]
        if type(index) is not int or index < 0 or index >= count or index in seen:
            raise ValueError("invalid_index")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("invalid_score")
        seen.add(index)
        scores[index] = float(score)
    return scores


async def rerank_async(query, hits, *, deadline, client=None):
    if not _slots.acquire(blocking=False):
        return hits, {"status": "fallback", "reason": "concurrency_limit"}
    try:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return hits, {"status": "fallback", "reason": "timeout"}
        # Keep a small request budget for latency, below Jina's 131k context.
        if len(query.encode("utf-8")) > 400:
            return hits, {"status": "fallback", "reason": "query_too_long"}
        inputs = [rerank_input(h, query) for h in hits]
        async with asyncio.timeout_at(asyncio.get_running_loop().time() + remaining):
            if client is None:
                if _client is None:
                    return hits, {"status": "fallback", "reason": "transport_not_ready"}
                client = _client
            response = await client.post(settings.rag_rerank_endpoint,
                headers={"Authorization": f"Bearer {settings.rag_rerank_api_key}"},
                json={"model": settings.rag_rerank_model, "query": query, "documents": [x[0] for x in inputs],
                      "top_n": len(hits), "return_documents": False})
            response.raise_for_status()
            payload = response.json()
            scores = validate_results(payload, len(hits))
        ranked = [{**h, "rerank_score": scores[i], "final_score": scores[i], "ranking_method": "model",
                   "rerank_input_truncated": inputs[i][1]} for i, h in enumerate(hits)]
        ranked.sort(key=lambda h: -h["rerank_score"])
        return ranked, {"status": "ok", "model": settings.rag_rerank_model,
                        "input_bytes": sum(len(x[0].encode()) for x in inputs),
                        "request_sha256": hashlib.sha256(json.dumps({"query": query, "documents": [x[0] for x in inputs]}, ensure_ascii=False).encode()).hexdigest(),
                        "response_id": payload.get("id") or response.headers.get("x-request-id"),
                        "usage": payload.get("usage")}
    except httpx.ConnectTimeout:
        return hits, {"status": "fallback", "reason": "connect_timeout"}
    except (TimeoutError, httpx.TimeoutException):
        return hits, {"status": "fallback", "reason": "timeout"}
    except httpx.HTTPStatusError as exc:
        return hits, {"status": "fallback", "reason": f"http_{exc.response.status_code}"}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return hits, {"status": "fallback", "reason": type(exc).__name__}
    finally:
        _slots.release()


def rerank_candidates(query, hits, user_id):
    started = perf_counter()
    if (_mode.get() or settings.rag_rerank_mode) != "model" or len(hits) < 2:
        return hits, {"status": "skipped", "reason": "disabled_or_single", "elapsed_ms": 0}
    allowed = {s.strip() for s in settings.rag_rerank_user_ids.split(",") if s.strip()}
    if allowed and str(user_id) not in allowed:
        return hits, {"status": "skipped", "reason": "user_not_enabled", "elapsed_ms": 0}
    if not settings.rag_rerank_api_key or not settings.rag_rerank_endpoint:
        return hits, {"status": "fallback", "reason": "not_configured", "elapsed_ms": 0}
    if len(hits) > 20:
        return hits, {"status": "fallback", "reason": "candidate_limit", "elapsed_ms": 0}
    preparation = prepare_reranker_transport(user_id)
    if preparation is None or not preparation.done() or preparation.exception() is not None:
        return hits, {"status": "fallback", "reason": "transport_not_ready",
                      "elapsed_ms": (perf_counter() - started) * 1000}
    deadline = started + settings.rag_rerank_timeout_ms / 1000
    future = asyncio.run_coroutine_threadsafe(rerank_async(query, hits, deadline=deadline), _loop)
    try:
        ranked, diagnostics = future.result(timeout=max(0, deadline - perf_counter()))
    except FutureTimeout:
        future.cancel()
        ranked, diagnostics = hits, {"status": "fallback", "reason": "timeout"}
    diagnostics["elapsed_ms"] = (perf_counter() - started) * 1000
    return ranked, diagnostics
