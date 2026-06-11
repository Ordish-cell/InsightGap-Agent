from pathlib import Path

import pytest

from scripts import run_rag_hybrid_eval as eval_runner
from scripts.compare_rag_backends import render_markdown, run_one
from src.web_app.db.repositories.document_repository import DocumentRepository
from src.web_app.models.orm import Document, User
from src.web_app.rag import embeddings
from src.web_app.tests.db_test_utils import make_test_session


class EvalFakeVectorStore:
    points: list[dict] = []
    hybrid_fail = False
    memory_delete_calls: list[str] = []

    def upsert_chunks(self, user_id, document_id, chunks, vectors, document):
        ids = []
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.get("metadata", {}))
            point_id = f"point-{document_id}-{index}"
            chunk_id = metadata.get("chunk_id") or point_id
            ids.append(point_id)
            self.points.append({
                "user_id": str(user_id),
                "document_id": str(document_id),
                "chunk_id": chunk_id,
                "qdrant_point_id": point_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "content_preview": chunk["content"][:200],
                "source_title": document.filename,
                "filename": document.filename,
                "file_type": document.file_type,
                "source_url": None,
                "metadata": metadata,
                "parent_id": metadata.get("parent_id"),
                "heading_path": chunk.get("heading_path", []),
                "score": 0.85,
            })
        return ids

    def search(self, user_id, query_vector, top_k=5, min_score=0.1, document_ids=None):
        rows = [row for row in self.points if row["user_id"] == str(user_id)]
        if document_ids:
            rows = [row for row in rows if int(row["document_id"]) in document_ids]
        return [{**row, "score": row.get("score", 0.85)} for row in rows[:top_k]]

    def search_hybrid(self, user_id, query_vector, query_text, top_k=5, min_score=0.1, document_ids=None):
        if self.hybrid_fail:
            raise RuntimeError("synthetic qdrant hybrid failed")
        rows = self.search(user_id, query_vector, top_k=top_k, min_score=min_score, document_ids=document_ids)
        return [{**row, "retrieval_source": "qdrant_hybrid", "final_score": row.get("score", 0.85)} for row in rows]

    def delete_document(self, user_id, document_id):
        type(self).points = [row for row in type(self).points if not (row["user_id"] == str(user_id) and row["document_id"] == str(document_id))]


@pytest.fixture()
def eval_env(monkeypatch):
    db = make_test_session()
    EvalFakeVectorStore.points = []
    EvalFakeVectorStore.hybrid_fail = False
    EvalFakeVectorStore.memory_delete_calls = []

    import src.web_app.services.document_service as document_service_module
    import src.web_app.services.rag_service as rag_service_module
    import src.web_app.rag.vector_store as vector_store_module

    monkeypatch.setattr(document_service_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(rag_service_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(rag_service_module.settings, "rag_hybrid_backend", "python_bm25")
    monkeypatch.setattr(document_service_module, "QdrantVectorStore", EvalFakeVectorStore)
    monkeypatch.setattr(rag_service_module, "QdrantVectorStore", EvalFakeVectorStore)
    monkeypatch.setattr(vector_store_module, "QdrantVectorStore", EvalFakeVectorStore)
    monkeypatch.setattr(document_service_module, "embed_texts", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(rag_service_module, "embed_text", lambda text: [0.1] * 384)
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(embeddings, "embed_text", lambda text: [0.1] * 384)
    yield db
    db.close()


def test_eval_fixtures_can_be_loaded():
    docs = eval_runner.load_fixture_documents()
    queries = eval_runner.load_queries()
    expected = eval_runner.load_expected()
    assert len(docs) >= 4
    assert any(path.name == "tech_agent_config.md" for path in docs)
    assert "合同编号是多少" in queries
    assert expected["合同编号是多少"] == ["HT-2026-001"]


def test_synthetic_documents_can_be_ingested(eval_env):
    user = eval_runner.get_or_create_test_user(eval_env)
    docs = eval_runner.ingest_fixtures(eval_env, user.id, force_reingest=True)
    eval_runner.validate_ingestion(eval_env, user.id, [doc["id"] for doc in docs])
    assert len(docs) >= 4
    assert EvalFakeVectorStore.points
    assert all((row["metadata"] or {}).get("chunk_role") == "child" for row in EvalFakeVectorStore.points)


def test_compare_outputs_two_backend_results(eval_env):
    user = eval_runner.get_or_create_test_user(eval_env)
    docs = eval_runner.ingest_fixtures(eval_env, user.id, force_reingest=True)
    records = [
        run_one(eval_env, user.id, "合同编号是多少", "python_bm25", 3, 0.1, [doc["id"] for doc in docs]),
        run_one(eval_env, user.id, "合同编号是多少", "qdrant_hybrid", 3, 0.1, [doc["id"] for doc in docs]),
    ]
    report = render_markdown(records)
    assert "python_bm25" in report
    assert "qdrant_hybrid" in report
    assert records[0]["results"]


def test_expected_keyword_metrics_work(eval_env):
    user = eval_runner.get_or_create_test_user(eval_env)
    docs = eval_runner.ingest_fixtures(eval_env, user.id, force_reingest=True)
    records = [
        run_one(eval_env, user.id, "合同编号是多少", backend, 5, 0.1, [doc["id"] for doc in docs])
        for backend in ("python_bm25", "qdrant_hybrid")
    ]
    metrics = eval_runner.compute_metrics(records, {"合同编号是多少": ["HT-2026-001"]}, top_k=5)
    assert {metric.backend for metric in metrics} == {"python_bm25", "qdrant_hybrid"}
    assert all(metric.total == 1 for metric in metrics)
    assert any(metric.keyword_hit_rate >= 0 for metric in metrics)


def test_qdrant_hybrid_fallback_is_recorded(eval_env):
    user = eval_runner.get_or_create_test_user(eval_env)
    docs = eval_runner.ingest_fixtures(eval_env, user.id, force_reingest=True)
    EvalFakeVectorStore.hybrid_fail = True
    record = run_one(eval_env, user.id, "合同编号是多少", "qdrant_hybrid", 3, 0.1, [doc["id"] for doc in docs])
    assert record["warning"]
    assert "failed" in record["warning"].lower()


def test_cleanup_only_removes_eval_documents(eval_env):
    user = eval_runner.get_or_create_test_user(eval_env)
    docs = eval_runner.ingest_fixtures(eval_env, user.id, force_reingest=True)
    real_doc = Document(user_id=user.id, filename="real-user-doc.txt", file_path="storage/uploads/fake/real.txt", file_type="txt", source_type="user_upload", status="uploaded", metadata_json={})
    eval_env.add(real_doc)
    eval_env.commit()
    eval_env.refresh(real_doc)

    eval_runner.cleanup_eval_documents(eval_env, user.id, [doc["id"] for doc in docs])
    assert DocumentRepository(eval_env).get_by_id_for_user(user.id, real_doc.id) is not None
    assert not EvalFakeVectorStore.memory_delete_calls


def test_cleanup_refuses_non_eval_document(eval_env):
    user = eval_runner.get_or_create_test_user(eval_env)
    real_doc = Document(user_id=user.id, filename="real-user-doc.txt", file_path="storage/uploads/fake/real.txt", file_type="txt", source_type="user_upload", status="uploaded", metadata_json={})
    eval_env.add(real_doc)
    eval_env.commit()
    eval_env.refresh(real_doc)
    with pytest.raises(RuntimeError):
        eval_runner.cleanup_eval_documents(eval_env, user.id, [real_doc.id])
