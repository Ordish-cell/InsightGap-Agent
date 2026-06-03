from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from src.web_app.api.v1.documents import get_current_user_id
from src.web_app.db.session import get_db
from src.web_app.main import app
from src.web_app.models.orm import User
from src.web_app.rag import embeddings
from src.web_app.rag import vector_store as vector_store_module
from src.web_app.tests.db_test_utils import make_test_session


class FakeVectorStore:
    points: list[dict] = []

    def ensure_collection(self):
        return None

    def upsert_chunks(self, user_id, document_id, chunks, vectors, document):
        ids = []
        for index, chunk in enumerate(chunks):
            point_id = f"point-{document_id}-{index}"
            ids.append(point_id)
            self.points.append(
                {
                    "user_id": str(user_id),
                    "document_id": str(document_id),
                    "chunk_id": point_id,
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "content_preview": chunk["content"][:200],
                    "source_title": document.filename,
                    "source_url": None,
                    "metadata": {},
                }
            )
        return ids

    def search(self, user_id, query_vector, top_k=5, min_score=0.2, document_ids=None):
        rows = [row for row in self.points if row["user_id"] == str(user_id)]
        if document_ids:
            rows = [row for row in rows if int(row["document_id"]) in document_ids]
        return [{**row, "score": 0.9} for row in rows[:top_k]]

    def delete_document(self, user_id, document_id):
        self.points = [row for row in self.points if not (row["user_id"] == str(user_id) and row["document_id"] == str(document_id))]

    def get_collection_stats(self):
        return {"collection": "fake", "points_count": len(self.points), "vectors_count": len(self.points)}


@pytest.fixture()
def client(monkeypatch):
    db = make_test_session()
    user = User(email="rag@example.com", hashed_password="x")
    other = User(email="other@example.com", hashed_password="x")
    db.add_all([user, other])
    db.commit()
    db.refresh(user)
    db.refresh(other)
    FakeVectorStore.points = []
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(embeddings, "embed_text", lambda text: [0.1] * 384)
    monkeypatch.setattr(vector_store_module, "QdrantVectorStore", FakeVectorStore)
    import src.web_app.services.document_service as document_service_module
    import src.web_app.services.rag_service as rag_service_module

    monkeypatch.setattr(document_service_module, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(document_service_module, "embed_texts", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(rag_service_module, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(rag_service_module, "embed_text", lambda text: [0.1] * 384)
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_id] = lambda: user.id
    with TestClient(app) as test_client:
        test_client.user_id = user.id
        test_client.other_user_id = other.id
        yield test_client
    app.dependency_overrides.clear()
    db.close()


def test_document_upload_txt_success(client):
    response = client.post("/api/v1/documents/upload", files={"file": ("note.txt", b"hello rag world", "text/plain")})
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "uploaded"


def test_upload_rejects_unsupported_file(client):
    response = client.post("/api/v1/documents/upload", files={"file": ("bad.exe", b"x", "application/octet-stream")})
    assert response.json()["success"] is False


def test_upload_prevents_path_traversal(client):
    response = client.post("/api/v1/documents/upload", files={"file": ("../safe.txt", b"safe", "text/plain")})
    assert response.status_code == 200
    assert response.json()["data"]["filename"] == "safe.txt"


def test_document_ingest_creates_chunks_and_writes_qdrant(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("rag.txt", b"# RAG\n\nQdrant stores evidence chunks for retrieval.", "text/plain")}).json()["data"]
    ingest = client.post(f"/api/v1/documents/{upload['id']}/ingest")
    assert ingest.status_code == 200
    assert ingest.json()["data"]["chunk_count"] >= 1
    assert FakeVectorStore.points


def test_rag_search_returns_evidence(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("search.txt", b"Agent OS uses evidence.", "text/plain")}).json()["data"]
    client.post(f"/api/v1/documents/{upload['id']}/ingest")
    response = client.post("/api/v1/rag/search", json={"query": "evidence", "top_k": 3, "min_score": 0.1})
    assert response.json()["data"]["results"]


def test_rag_ask_returns_answer_with_evidence(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("ask.txt", b"Evidence based answers cite chunks.", "text/plain")}).json()["data"]
    client.post(f"/api/v1/documents/{upload['id']}/ingest")
    response = client.post("/api/v1/rag/ask", json={"question": "How answer?", "top_k": 3, "min_score": 0.1})
    data = response.json()["data"]
    assert data["evidence"]
    assert data["answer_mode"] == "extractive_fallback"


def test_rag_ask_no_evidence_does_not_hallucinate(client):
    response = client.post("/api/v1/rag/ask", json={"question": "unknown", "top_k": 3, "min_score": 0.1})
    data = response.json()["data"]
    assert data["evidence"] == []
    assert "没有找到足够证据" in data["answer"]


def test_user_cannot_access_other_user_document(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("mine.txt", b"mine", "text/plain")}).json()["data"]
    app.dependency_overrides[get_current_user_id] = lambda: client.other_user_id
    response = client.get(f"/api/v1/documents/{upload['id']}")
    assert response.json()["success"] is False


def test_qdrant_search_filters_by_user_id(client):
    FakeVectorStore.points.append({"user_id": "9999", "document_id": "99", "chunk_id": "x", "chunk_index": 0, "content": "secret", "content_preview": "secret", "source_title": "x", "source_url": None, "metadata": {}})
    response = client.post("/api/v1/rag/search", json={"query": "secret", "top_k": 5, "min_score": 0.1})
    assert response.json()["data"]["results"] == []


def test_health_dependencies_includes_embedding_and_qdrant(client):
    response = client.get("/api/v1/health/dependencies")
    data = response.json()["data"]
    assert "embedding_provider" in data
    assert "qdrant_collection" in data
