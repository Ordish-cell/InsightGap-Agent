import logging
from io import BytesIO

import pytest
import requests
from fastapi.testclient import TestClient

from src.web_app.api.v1.documents import get_current_user_id
from src.web_app.db.session import get_db
from src.web_app.main import app
from src.web_app.models.orm import User
from src.web_app.rag import embeddings
from src.web_app.services.document_service import DocumentService
from src.web_app.tests.db_test_utils import make_test_session


class DashScopeResponse:
    def __init__(self, status_code=200, text="{}", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload if payload is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


def test_dashscope_http_error_logs_body_and_safe_context(monkeypatch, caplog):
    monkeypatch.setattr(embeddings.settings, "embed_api_key", "test-key")
    monkeypatch.setattr(embeddings.settings, "embed_base_url", "https://dashscope.test/v1")
    monkeypatch.setattr(embeddings.settings, "dashscope_embedding_batch_size", 10)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: {"provider": "dashscope", "model": "text-embedding-v4", "tier": "low"})
    monkeypatch.setattr(embeddings.requests, "post", lambda *args, **kwargs: DashScopeResponse(400, '{"code":"InvalidParameter","message":"bad input"}'))

    caplog.set_level(logging.ERROR, logger="src.web_app.rag.embeddings")
    with pytest.raises(requests.HTTPError):
        embeddings._embed_dashscope(["alpha content", "beta content"])

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "status_code=400" in logs
    assert "InvalidParameter" in logs
    assert "model=text-embedding-v4" in logs
    assert "input_count=2" in logs
    assert "alpha content" in logs


def test_dashscope_batches_25_inputs_as_10_10_5_and_preserves_order(monkeypatch):
    calls = []

    def post(_url, headers, json, timeout):
        del headers, timeout
        batch_texts = json["input"]
        calls.append(list(batch_texts))
        data = []
        for index, text in enumerate(batch_texts):
            value = float(text.removeprefix("text-"))
            data.append({"index": index, "embedding": [value]})
        return DashScopeResponse(payload={"data": data})

    monkeypatch.setattr(embeddings.settings, "embed_api_key", "test-key")
    monkeypatch.setattr(embeddings.settings, "embed_base_url", "https://dashscope.test/v1")
    monkeypatch.setattr(embeddings.settings, "dashscope_embedding_batch_size", 10)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: {"provider": "dashscope", "model": "text-embedding-v4", "tier": "low"})
    monkeypatch.setattr(embeddings.requests, "post", post)

    vectors = embeddings._embed_dashscope([f"text-{index}" for index in range(25)])

    assert [len(call) for call in calls] == [10, 10, 5]
    assert [vector[0] for vector in vectors] == [float(index) for index in range(25)]


def test_dashscope_batch_size_clamps_to_10(monkeypatch, caplog):
    calls = []

    def post(_url, headers, json, timeout):
        del headers, timeout
        calls.append(list(json["input"]))
        return DashScopeResponse(payload={"data": [{"index": index, "embedding": [0.1]} for index, _ in enumerate(json["input"])]})

    monkeypatch.setattr(embeddings.settings, "embed_api_key", "test-key")
    monkeypatch.setattr(embeddings.settings, "embed_base_url", "https://dashscope.test/v1")
    monkeypatch.setattr(embeddings.settings, "dashscope_embedding_batch_size", 99)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: {"provider": "dashscope", "model": "text-embedding-v4", "tier": "low"})
    monkeypatch.setattr(embeddings.requests, "post", post)

    caplog.set_level(logging.WARNING, logger="src.web_app.rag.embeddings")
    embeddings._embed_dashscope([f"text-{index}" for index in range(12)])

    assert [len(call) for call in calls] == [10, 2]
    assert "batch_size_clamped" in "\n".join(record.getMessage() for record in caplog.records)


def test_dashscope_failed_batch_logs_batch_context(monkeypatch, caplog):
    def post(_url, headers, json, timeout):
        del headers, timeout
        if json["input"][0] == "text-10":
            return DashScopeResponse(400, '{"code":"InvalidParameter","message":"batch failed"}')
        return DashScopeResponse(payload={"data": [{"index": index, "embedding": [0.1]} for index, _ in enumerate(json["input"])]})

    monkeypatch.setattr(embeddings.settings, "embed_api_key", "test-key")
    monkeypatch.setattr(embeddings.settings, "embed_base_url", "https://dashscope.test/v1")
    monkeypatch.setattr(embeddings.settings, "dashscope_embedding_batch_size", 10)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: {"provider": "dashscope", "model": "text-embedding-v4", "tier": "low"})
    monkeypatch.setattr(embeddings.requests, "post", post)

    caplog.set_level(logging.ERROR, logger="src.web_app.rag.embeddings")
    with pytest.raises(requests.HTTPError):
        embeddings._embed_dashscope([f"text-{index}" for index in range(12)])

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "batch_index=1" in logs
    assert "batch_start=10" in logs
    assert "batch_end=12" in logs
    assert "batch_size=2" in logs
    assert "status_code=400" in logs
    assert "batch failed" in logs
    assert "model=text-embedding-v4" in logs
    assert "text-10" in logs


def test_dashscope_vector_count_mismatch_raises(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "embed_api_key", "test-key")
    monkeypatch.setattr(embeddings.settings, "embed_base_url", "https://dashscope.test/v1")
    monkeypatch.setattr(embeddings.settings, "dashscope_embedding_batch_size", 10)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: {"provider": "dashscope", "model": "text-embedding-v4", "tier": "low"})
    monkeypatch.setattr(embeddings.requests, "post", lambda *args, **kwargs: DashScopeResponse(payload={"data": [{"index": 0, "embedding": [0.1]}]}))

    with pytest.raises(RuntimeError, match="unexpected number of vectors"):
        embeddings._embed_dashscope(["one", "two"])


def test_embed_texts_empty_input_returns_empty_list():
    assert embeddings.embed_texts([]) == []
    assert embeddings.embed_texts(["", None]) == []


def test_embed_texts_filters_empty_inputs(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: {"provider": "dashscope", "model": "text-embedding-v4", "tier": "low"})
    monkeypatch.setattr(embeddings, "_embed_dashscope", lambda texts: [[0.1] * embeddings.get_embedding_dimension() for _ in texts])
    vectors = embeddings.embed_texts(["", None, " valid "])
    assert len(vectors) == 1


def test_document_service_rejects_too_long_embedding_chunk(caplog):
    service = DocumentService()
    caplog.set_level(logging.ERROR, logger="src.web_app.services.document_service")
    long_text = "x" * (embeddings.MAX_EMBED_CHARS + 1)
    with pytest.raises(ValueError):
        service._validate_embedding_chunks(123, [{"chunk_index": 7, "content": long_text, "metadata": {"chunk_id": "c-too-long"}}])
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "document_id=123" in logs
    assert "chunk_index=7" in logs
    assert "chunk_id=c-too-long" in logs


def test_chat_upload_ingest_failure_response_has_failed_status(monkeypatch):
    db = make_test_session()
    user = User(email="embed-fail@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_id] = lambda: user.id

    import src.web_app.services.document_service as document_service_module

    def fail_embed(_texts):
        raise requests.HTTPError("400 Client Error: Bad Request for url")

    monkeypatch.setattr(document_service_module.settings, "qdrant_url", "")
    monkeypatch.setattr(document_service_module, "embed_texts", fail_embed)
    with TestClient(app) as client:
        response = client.post("/api/v1/documents/chat-upload", files={"file": ("bad.txt", b"hello", "text/plain")})
    data = response.json()
    app.dependency_overrides.clear()
    db.close()

    assert data["success"] is False
    assert data["error"]["details"]["ingest_status"] == "failed"
    assert data["error"]["details"]["error_message"]
