from types import SimpleNamespace

import pytest
from qdrant_client import models

from src.web_app.db.session import get_db
from src.web_app.main import app
from src.web_app.models.orm import Document, DocumentChunk, User
from src.web_app.rag import sparse_encoder
from src.web_app.rag.vector_store import QdrantVectorStore
from src.web_app.services.rag_service import rag_service
from src.web_app.tests.db_test_utils import make_test_session


class FakeHybridVectorStore:
    should_fail = False
    hybrid_hits: list[dict] = []

    def search(self, user_id, query_vector, top_k=5, min_score=0.2, document_ids=None):
        return []

    def search_hybrid(self, user_id, query_vector, query_text, top_k=5, min_score=0.2, document_ids=None):
        if self.should_fail:
            raise RuntimeError("native hybrid failed")
        rows = [row for row in self.hybrid_hits if row["user_id"] == str(user_id)]
        if document_ids:
            rows = [row for row in rows if int(row["document_id"]) in document_ids]
        return [{**row, "retrieval_source": "qdrant_hybrid", "final_score": row.get("score", 0.7)} for row in rows[:top_k]]

    def capability_status(self):
        return {"supported": not self.should_fail}


@pytest.fixture()
def qdrant_hybrid_env(monkeypatch):
    db = make_test_session()
    user = User(email="qdrant-hybrid@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    FakeHybridVectorStore.should_fail = False
    FakeHybridVectorStore.hybrid_hits = []

    import src.web_app.services.rag_service as rag_service_module
    monkeypatch.setattr(rag_service_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(rag_service_module.settings, "rag_hybrid_backend", "qdrant_hybrid")
    monkeypatch.setattr(rag_service_module.settings, "qdrant_hybrid_fallback", True)
    monkeypatch.setattr(rag_service_module, "QdrantVectorStore", FakeHybridVectorStore)
    monkeypatch.setattr(rag_service_module, "embed_text", lambda _text: [0.1] * 384)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield db, user
    finally:
        app.dependency_overrides.clear()
        db.close()


def test_qdrant_hybrid_child_hit_enriches_parent_context(qdrant_hybrid_env):
    db, user = qdrant_hybrid_env
    doc, child = _add_document_with_parent_child(db, user.id, "native.md", "native child content")
    FakeHybridVectorStore.hybrid_hits.append(_hit(user.id, doc.id, child.metadata_json["chunk_id"], child.content, child.metadata_json["parent_id"], doc.filename))

    result = rag_service.search(user.id, "native", top_k=3, min_score=0.1, db=db)["results"][0]
    assert result["retrieval_source"] == "qdrant_hybrid"
    assert result["parent_context_available"] is True
    assert "Parent native.md" in result["parent_context"]
    assert result["citation"]["child_chunk_id"] == result["chunk_id"]


def test_qdrant_hybrid_failure_falls_back_to_python_bm25(qdrant_hybrid_env):
    db, user = qdrant_hybrid_env
    _add_document_with_parent_child(db, user.id, "fallback.md", "contract number HT-QDRANT-FALLBACK")
    FakeHybridVectorStore.should_fail = True

    result = rag_service.search(user.id, "HT-QDRANT-FALLBACK", top_k=3, min_score=0.1, db=db)
    assert result["results"]
    assert result["results"][0]["retrieval_source"] == "bm25"
    assert "qdrant_hybrid_failed" in result["retrieval_warning"]


def test_qdrant_hybrid_search_old_and_new_fields(qdrant_hybrid_env):
    db, user = qdrant_hybrid_env
    doc, child = _add_document_with_parent_child(db, user.id, "compat-native.md", "compat native content")
    FakeHybridVectorStore.hybrid_hits.append(_hit(user.id, doc.id, child.metadata_json["chunk_id"], child.content, child.metadata_json["parent_id"], doc.filename))

    result = rag_service.search(user.id, "compat", top_k=3, min_score=0.1, db=db)["results"][0]
    for field in ("chunk_id", "content", "score", "source_title", "metadata"):
        assert field in result
    for field in ("retrieval_source", "final_score", "query_type", "parent_context", "citation"):
        assert field in result


def test_qdrant_hybrid_no_sparse_encoder_falls_back(qdrant_hybrid_env, monkeypatch):
    db, user = qdrant_hybrid_env
    _add_document_with_parent_child(db, user.id, "sparse-missing.md", "sparse missing HT-SPARSE")

    class EmptySparseStore(FakeHybridVectorStore):
        def search_hybrid(self, *args, **kwargs):
            raise RuntimeError("Sparse query vector is empty")

    import src.web_app.services.rag_service as rag_service_module
    monkeypatch.setattr(rag_service_module, "QdrantVectorStore", EmptySparseStore)
    result = rag_service.search(user.id, "HT-SPARSE", top_k=3, min_score=0.1, db=db)
    assert result["results"][0]["retrieval_source"] == "bm25"
    assert "Sparse query vector is empty" in result["retrieval_warning"]


def test_sparse_input_factory_uses_qdrant_cloud_bm25_document(monkeypatch):
    monkeypatch.setattr(sparse_encoder.settings, "qdrant_sparse_encoder", "qdrant_cloud_bm25")
    monkeypatch.setattr(sparse_encoder.settings, "qdrant_sparse_model", "Qdrant/bm25")

    value = sparse_encoder.build_sparse_document_input("official bm25 text")

    assert isinstance(value, models.Document)
    assert value.text == "official bm25 text"
    assert value.model == "Qdrant/bm25"


def test_sparse_input_factory_hashing_fallback_returns_sparse_vector(monkeypatch):
    monkeypatch.setattr(sparse_encoder.settings, "qdrant_sparse_encoder", "hashing_sparse")

    value = sparse_encoder.build_sparse_query_input("contract HT-2026-001")

    assert isinstance(value, models.SparseVector)
    assert value.indices
    assert value.values


def test_qdrant_hybrid_schema_uses_v3_collection(monkeypatch):
    fake_client = FakeQdrantClient(collections=[])
    import src.web_app.rag.vector_store as vector_store_module
    monkeypatch.setattr(vector_store_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(vector_store_module.settings, "rag_hybrid_backend", "qdrant_hybrid")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_collection", "agent_os_documents")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_hybrid_collection", "agent_os_documents_v3")
    monkeypatch.setattr(vector_store_module, "QdrantClient", lambda **_kwargs: fake_client)

    store = QdrantVectorStore()
    store.ensure_collection()

    assert fake_client.created_collection == "agent_os_documents_v3"
    assert "dense" in fake_client.created_vectors_config
    assert "bm25" in fake_client.created_sparse_vectors_config
    assert fake_client.created_collection != "agent_os_documents"


def test_qdrant_hybrid_capability_rejects_dense_only_collection(monkeypatch):
    fake_client = FakeQdrantClient(collections=["agent_os_documents_v3"], dense_only=True)
    import src.web_app.rag.vector_store as vector_store_module
    monkeypatch.setattr(vector_store_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(vector_store_module.settings, "rag_hybrid_backend", "qdrant_hybrid")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_hybrid_collection", "agent_os_documents_v3")
    monkeypatch.setattr(vector_store_module, "QdrantClient", lambda **_kwargs: fake_client)

    store = QdrantVectorStore()
    status = store.capability_status()
    assert status["supported"] is False
    assert "dense_sparse_collection_schema" in status["missing"]


def test_qdrant_hybrid_upsert_writes_dense_and_sparse_vectors(monkeypatch):
    fake_client = FakeQdrantClient(collections=[])
    import src.web_app.rag.vector_store as vector_store_module
    monkeypatch.setattr(vector_store_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(vector_store_module.settings, "rag_hybrid_backend", "qdrant_hybrid")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_sparse_encoder", "qdrant_cloud_bm25")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_sparse_model", "Qdrant/bm25")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_cloud_inference", True)
    monkeypatch.setattr(vector_store_module, "QdrantClient", lambda **_kwargs: fake_client)
    document = SimpleNamespace(filename="child.md", file_type="md")
    chunk = {"chunk_index": 1, "content": "contract HT-100", "token_count": 4, "heading_path": [], "metadata": {"chunk_role": "child", "chunk_id": "c-1", "parent_id": "p-1"}}

    QdrantVectorStore().upsert_chunks(1, 2, [chunk], [[0.1] * 384], document)

    point = fake_client.upserted_points[0]
    assert "dense" in point.vector
    assert "bm25" in point.vector
    assert isinstance(point.vector["bm25"], models.Document)
    assert point.vector["bm25"].model == "Qdrant/bm25"
    assert point.payload["chunk_role"] == "child"
    assert point.payload["parent_id"] == "p-1"


def test_qdrant_client_enables_cloud_inference(monkeypatch):
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeQdrantClient(collections=[])

    import src.web_app.rag.vector_store as vector_store_module
    monkeypatch.setattr(vector_store_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_cloud_inference", True)
    monkeypatch.setattr(vector_store_module, "QdrantClient", client_factory)

    QdrantVectorStore()

    assert captured["cloud_inference"] is True


def test_qdrant_cloud_bm25_success_does_not_call_python_bm25(qdrant_hybrid_env, monkeypatch):
    db, user = qdrant_hybrid_env
    doc, child = _add_document_with_parent_child(db, user.id, "no-bm25.md", "official cloud bm25 child")
    FakeHybridVectorStore.hybrid_hits.append(_hit(user.id, doc.id, child.metadata_json["chunk_id"], child.content, child.metadata_json["parent_id"], doc.filename))

    import src.web_app.db.repositories.document_repository as document_repo_module

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Python BM25 candidate loader should not be called when Qdrant hybrid succeeds")

    monkeypatch.setattr(document_repo_module.DocumentChunkRepository, "list_child_candidates", fail_if_called)

    result = rag_service.search(user.id, "official cloud bm25", top_k=3, min_score=0.1, db=db)
    assert result["results"][0]["retrieval_source"] == "qdrant_hybrid"


def test_qdrant_cloud_bm25_upsert_failure_does_not_write_hashing_sparse(monkeypatch):
    fake_client = FakeQdrantClient(collections=[], fail_upsert=True)
    import src.web_app.rag.vector_store as vector_store_module
    monkeypatch.setattr(vector_store_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(vector_store_module.settings, "rag_hybrid_backend", "qdrant_hybrid")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_sparse_encoder", "qdrant_cloud_bm25")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_sparse_model", "Qdrant/bm25")
    monkeypatch.setattr(vector_store_module, "QdrantClient", lambda **_kwargs: fake_client)
    document = SimpleNamespace(filename="child.md", file_type="md")
    chunk = {"chunk_index": 1, "content": "cloud inference unavailable", "token_count": 4, "heading_path": [], "metadata": {"chunk_role": "child", "chunk_id": "c-1", "parent_id": "p-1"}}

    with pytest.raises(RuntimeError, match="cloud inference unavailable"):
        QdrantVectorStore().upsert_chunks(1, 2, [chunk], [[0.1] * 384], document)

    point = fake_client.last_points[0]
    assert isinstance(point.vector["bm25"], models.Document)


def test_delete_document_missing_hybrid_collection_is_noop(monkeypatch):
    fake_client = FakeQdrantClient(collections=[])
    import src.web_app.rag.vector_store as vector_store_module
    monkeypatch.setattr(vector_store_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(vector_store_module.settings, "rag_hybrid_backend", "qdrant_hybrid")
    monkeypatch.setattr(vector_store_module.settings, "qdrant_hybrid_collection", "agent_os_documents_v3")
    monkeypatch.setattr(vector_store_module, "QdrantClient", lambda **_kwargs: fake_client)

    QdrantVectorStore().delete_document(1, 2)

    assert fake_client.delete_calls == []
    assert fake_client.created_collection is None


class FakeQdrantClient:
    def __init__(self, collections=None, dense_only=False, fail_upsert=False):
        self.collections = collections or []
        self.dense_only = dense_only
        self.fail_upsert = fail_upsert
        self.created_collection = None
        self.created_vectors_config = {}
        self.created_sparse_vectors_config = {}
        self.upserted_points = []
        self.last_points = []
        self.delete_calls = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    def create_collection(self, collection_name, vectors_config=None, sparse_vectors_config=None, **_kwargs):
        self.created_collection = collection_name
        self.collections.append(collection_name)
        self.created_vectors_config = vectors_config or {}
        self.created_sparse_vectors_config = sparse_vectors_config or {}
        return True

    def get_collection(self, collection_name):
        if self.dense_only:
            vectors = {"dense": object()}
            sparse_vectors = {}
        else:
            vectors = {"dense": object()}
            sparse_vectors = {"bm25": object()}
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors, sparse_vectors=sparse_vectors)), points_count=0)

    def create_payload_index(self, **_kwargs):
        return True

    def upsert(self, collection_name, points):
        self.last_points = list(points)
        if self.fail_upsert:
            raise RuntimeError("cloud inference unavailable")
        self.upserted_points.extend(points)
        return True

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)
        return True


def _add_document_with_parent_child(db, user_id: int, filename: str, child_content: str):
    document = Document(user_id=user_id, filename=filename, file_path=f"storage/uploads/fake/{filename}", file_type="md", source_type="user_upload", status="ingested", metadata_json={})
    db.add(document)
    db.commit()
    db.refresh(document)
    parent_id = "p-native"
    parent = DocumentChunk(user_id=user_id, document_id=document.id, chunk_index=0, content=f"# Parent {filename}\n\n{child_content}", token_count=10, qdrant_point_id="", metadata_json={"chunk_role": "parent", "chunk_id": parent_id})
    child = DocumentChunk(user_id=user_id, document_id=document.id, chunk_index=1, content=child_content, token_count=10, qdrant_point_id=f"point-{filename}", metadata_json={"chunk_role": "child", "chunk_id": f"c-{document.id}", "parent_id": parent_id, "chunk_type": "section"})
    db.add_all([parent, child])
    db.commit()
    db.refresh(child)
    return document, child


def _hit(user_id: int, document_id: int, chunk_id: str, content: str, parent_id: str, filename: str):
    return {
        "user_id": str(user_id),
        "document_id": str(document_id),
        "chunk_id": chunk_id,
        "chunk_index": 1,
        "score": 0.72,
        "content": content,
        "content_preview": content[:200],
        "source_title": filename,
        "source_url": None,
        "filename": filename,
        "metadata": {"chunk_role": "child", "chunk_id": chunk_id, "parent_id": parent_id},
        "parent_id": parent_id,
    }
