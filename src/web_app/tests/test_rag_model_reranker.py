import asyncio
import time
import httpx
import pytest
from src.web_app.rag.model_reranker import rerank_async, rerank_candidates, rerank_input, validate_results
from src.web_app.core.config import settings


@pytest.mark.parametrize("rows", [[], [{"index": 0, "relevance_score": 1}] * 2,
    [{"index": -1, "relevance_score": 1}, {"index": 1, "relevance_score": 1}],
    [{"index": 0, "relevance_score": float('nan')}, {"index": 1, "relevance_score": 1}]])
def test_invalid_results_rejected(rows):
    with pytest.raises(ValueError):
        validate_results({"results": rows}, 2)


def test_truncation_keeps_late_query_match():
    text, truncated = rerank_input({"content": "开头" * 2000 + " ZX-912 是目标"}, "ZX-912")
    assert truncated and "ZX-912" in text
    assert len(text.encode()) <= 1000


def test_success_and_tie_order(monkeypatch):
    monkeypatch.setattr(settings, "rag_rerank_endpoint", "https://rerank.test/api")
    hits = [{"content": "a", "score": .9}, {"content": "b", "score": .5}]
    async def run():
        def handle(request):
            return httpx.Response(200, json={"results": [
                {"index": 1, "relevance_score": .8}, {"index": 0, "relevance_score": .1}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            ranked, info = await rerank_async("q", hits, deadline=time.perf_counter() + .9, client=client)
        assert ranked[0]["content"] == "b" and ranked[0]["score"] == .5
        assert info["status"] == "ok"
    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_fallback_atomically(monkeypatch, status):
    monkeypatch.setattr(settings, "rag_rerank_endpoint", "https://rerank.test/api")
    hits = [{"content": "a"}, {"content": "b"}]
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status))) as client:
            ranked, info = await rerank_async("q", hits, deadline=time.perf_counter() + .9, client=client)
        assert ranked is hits
        assert info["reason"] == f"http_{status}"
    asyncio.run(run())


def test_deadline_cancels_http(monkeypatch):
    monkeypatch.setattr(settings, "rag_rerank_endpoint", "https://rerank.test/api")
    cancelled = []
    async def handle(request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _, info = await rerank_async("q", [{"content": "a"}], deadline=time.perf_counter() + .03, client=client)
        assert info["reason"] == "timeout"
        assert cancelled
    asyncio.run(run())


def test_disabled_needs_no_config(monkeypatch):
    monkeypatch.setattr(settings, "rag_rerank_mode", "off")
    hits = [{"content": "a"}, {"content": "b"}]
    assert rerank_candidates("q", hits, 1)[0] is hits


def test_connect_timeout_has_distinct_diagnostic(monkeypatch):
    monkeypatch.setattr(settings, "rag_rerank_endpoint", "https://rerank.test/api")
    async def run():
        def handle(request):
            raise httpx.ConnectTimeout("test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _, diagnostic = await rerank_async("q", [{"content": "a"}], deadline=time.perf_counter() + .9, client=client)
        assert diagnostic['reason'] == 'connect_timeout'
    asyncio.run(run())


def test_unprepared_client_never_constructed_inside_budget(monkeypatch):
    import src.web_app.rag.model_reranker as module
    monkeypatch.setattr(module, '_client', None)
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: pytest.fail('constructed on critical path'))
    result, diagnostic = asyncio.run(rerank_async('q', [{'content': 'a'}], deadline=time.perf_counter() + .9))
    assert diagnostic['reason'] == 'transport_not_ready'


def test_warming_returns_without_waiting(monkeypatch):
    from concurrent.futures import Future
    import src.web_app.rag.model_reranker as module
    monkeypatch.setattr(settings, 'rag_rerank_mode', 'model')
    monkeypatch.setattr(settings, 'rag_rerank_user_ids', '')
    monkeypatch.setattr(settings, 'rag_rerank_api_key', 'test-only')
    monkeypatch.setattr(settings, 'rag_rerank_endpoint', 'https://rerank.test/api')
    monkeypatch.setattr(module, 'prepare_reranker_transport', lambda user_id: Future())
    _, diagnostic = rerank_candidates('q', [{'content': 'a'}, {'content': 'b'}], 1)
    assert diagnostic['reason'] == 'transport_not_ready'
    assert diagnostic['elapsed_ms'] < 100


def test_transport_configuration_and_single_certificate_context(monkeypatch):
    import src.web_app.rag.model_reranker as module
    captured = {}
    sentinel = object()
    monkeypatch.setattr(module, '_client', None)
    monkeypatch.setattr(module.ssl, 'create_default_context', lambda **kwargs: sentinel)
    def client(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(httpx, 'AsyncClient', client)
    asyncio.run(module._initialize_client())
    assert captured['verify'] is sentinel
    assert captured['timeout'].connect == .6
    assert captured['timeout'].read == .9
    assert captured['limits'].keepalive_expiry == 120
