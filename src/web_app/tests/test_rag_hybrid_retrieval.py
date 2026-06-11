import pytest

from src.web_app.db.session import get_db
from src.web_app.main import app
from src.web_app.models.orm import Document, DocumentChunk, User
from src.web_app.rag.bm25 import BM25Document, bm25_search, tokenize
from src.web_app.rag.query_analyzer import analyze_query
from src.web_app.services.rag_service import rag_service
from src.web_app.tests.db_test_utils import make_test_session


class FakeVectorStore:
    points: list[dict] = []

    def search(self, user_id, query_vector, top_k=5, min_score=0.2, document_ids=None):
        rows = [row for row in self.points if row["user_id"] == str(user_id)]
        if document_ids:
            rows = [row for row in rows if int(row["document_id"]) in document_ids]
        return [{**row, "score": row.get("score", 0.9)} for row in rows[:top_k]]


@pytest.fixture()
def hybrid_env(monkeypatch):
    db = make_test_session()
    user = User(email="hybrid@example.com", hashed_password="x")
    other = User(email="other-hybrid@example.com", hashed_password="x")
    db.add_all([user, other])
    db.commit()
    db.refresh(user)
    db.refresh(other)
    FakeVectorStore.points = []

    import src.web_app.services.rag_service as rag_service_module
    monkeypatch.setattr(rag_service_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(rag_service_module, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(rag_service_module, "embed_text", lambda _text: [0.1] * 384)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield db, user, other
    finally:
        app.dependency_overrides.clear()
        db.close()


def test_bm25_tokenizer_preserves_exact_tokens():
    tokens = tokenize("合同号 HT-2026-001 email finance@example.com amount 12345.67 parse_invoice()")
    assert "ht-2026-001" in tokens
    assert "finance@example.com" in tokens
    assert "12345.67" in tokens
    assert "parse_invoice()" in tokens


def test_bm25_chinese_ngram_hits_chinese_content():
    hits = bm25_search("风险控制", [BM25Document(id="a", content="本文讨论供应链风险控制策略。")], top_k=3)
    assert hits
    assert hits[0].matched_terms


def test_query_analyzer_detects_query_types():
    assert analyze_query("总结这个文档").query_type == "summary"
    assert analyze_query("合同号 HT-2026-001 是多少").query_type == "exact"
    assert analyze_query("sheet 里 amount 列是哪一行").query_type == "table"
    assert analyze_query("为什么会有这个风险").query_type == "semantic"


def test_hybrid_marks_vector_and_bm25_same_child_as_hybrid(hybrid_env):
    db, user, _ = hybrid_env
    doc, child = _add_document_with_child(db, user.id, "hybrid.md", "The contract number is HT-2026-001.", chunk_id="c-hybrid")
    FakeVectorStore.points.append(_vector_point(user.id, doc.id, child.metadata_json["chunk_id"], child.content, score=0.8, parent_id=child.metadata_json["parent_id"], filename=doc.filename))

    result = rag_service.search(user.id, "HT-2026-001", top_k=5, min_score=0.1, db=db)["results"][0]
    assert result["retrieval_source"] == "hybrid"
    assert result["vector_score"] > 0
    assert result["bm25_score"] > 0
    assert "ht-2026-001" in result["matched_terms"]


def test_bm25_only_result_when_vector_misses(hybrid_env):
    db, user, _ = hybrid_env
    doc, child = _add_document_with_child(db, user.id, "bm25.md", "Contact finance@example.com for invoices.", chunk_id="c-email")
    FakeVectorStore.points.append(_vector_point(user.id, doc.id, "unrelated", "unrelated semantic result", score=0.1, parent_id="missing", filename=doc.filename))

    results = rag_service.search(user.id, "finance@example.com", top_k=5, min_score=0.1, db=db)["results"]
    email_hit = next(item for item in results if item["chunk_id"] == child.metadata_json["chunk_id"])
    assert email_hit["retrieval_source"] == "bm25"
    assert email_hit["bm25_score"] > 0


def test_exact_query_promotes_bm25_hit(hybrid_env):
    db, user, _ = hybrid_env
    doc, _ = _add_document_with_child(db, user.id, "exact.md", "General semantic paragraph.", chunk_id="c-general")
    _add_child(db, user.id, doc.id, "产品型号 XR-9000-Pro 出现在这里。", chunk_id="c-model", parent_id="p-exact")
    FakeVectorStore.points.append(_vector_point(user.id, doc.id, "c-general", "General semantic paragraph.", score=0.95, parent_id="p-exact", filename=doc.filename))

    results = rag_service.search(user.id, "产品型号 XR-9000-Pro", top_k=5, min_score=0.1, db=db)["results"]
    assert results[0]["chunk_id"] == "c-model"
    assert results[0]["retrieval_source"] == "bm25"


def test_semantic_query_keeps_vector_advantage(hybrid_env):
    db, user, _ = hybrid_env
    doc, child = _add_document_with_child(db, user.id, "semantic.md", "Risk mitigation requires staged rollout.", chunk_id="c-vector")
    _add_child(db, user.id, doc.id, "invoice number ZZ-1", chunk_id="c-keyword", parent_id="p-semantic")
    FakeVectorStore.points.append(_vector_point(user.id, doc.id, child.metadata_json["chunk_id"], child.content, score=0.95, parent_id=child.metadata_json["parent_id"], filename=doc.filename))

    result = rag_service.search(user.id, "why risk mitigation matters", top_k=5, min_score=0.1, db=db)["results"][0]
    assert result["chunk_id"] == "c-vector"
    assert result["retrieval_source"] in {"vector", "hybrid"}


def test_table_query_boosts_row_block(hybrid_env):
    db, user, _ = hybrid_env
    doc, child = _add_document_with_child(
        db,
        user.id,
        "sheet.csv",
        "Columns: team | amount\nRow 1: team: infra; amount: 120",
        chunk_id="c-row",
        metadata={"chunk_type": "row_block", "sheet_name": "Budget", "header": ["team", "amount"]},
    )
    result = rag_service.search(user.id, "sheet amount column", top_k=5, min_score=0.1, db=db)["results"][0]
    assert result["chunk_id"] == child.metadata_json["chunk_id"]
    assert result["query_type"] == "table"
    assert result["final_score"] > 0
    assert result["metadata"]["chunk_type"] == "row_block"


def test_same_parent_multiple_children_get_parent_boost(hybrid_env):
    db, user, _ = hybrid_env
    doc, _ = _add_document_with_child(db, user.id, "parent.md", "alpha keyword first child", chunk_id="c-parent-1", parent_id="p-shared")
    _add_child(db, user.id, doc.id, "alpha keyword second child", chunk_id="c-parent-2", parent_id="p-shared")
    results = rag_service.search(user.id, "alpha keyword", top_k=5, min_score=0.1, db=db)["results"]
    assert len([item for item in results if item["parent_id"] == "p-shared"]) == 2
    assert all(item["final_score"] > 0 for item in results if item["parent_id"] == "p-shared")


def test_hybrid_respects_user_and_document_filters(hybrid_env):
    db, user, other = hybrid_env
    allowed_doc, _ = _add_document_with_child(db, user.id, "allowed.md", "allowed contract HT-ALLOW", chunk_id="c-allow")
    blocked_doc, _ = _add_document_with_child(db, user.id, "blocked.md", "blocked contract HT-BLOCK", chunk_id="c-block")
    _add_document_with_child(db, other.id, "other.md", "other contract HT-OTHER", chunk_id="c-other")

    results = rag_service.search(user.id, "contract HT", top_k=10, min_score=0.1, document_ids=[allowed_doc.id], db=db)["results"]
    assert results
    assert {int(item["document_id"]) for item in results} == {allowed_doc.id}
    assert blocked_doc.id not in {int(item["document_id"]) for item in results}


def test_search_old_fields_and_new_fields_are_present(hybrid_env):
    db, user, _ = hybrid_env
    doc, child = _add_document_with_child(db, user.id, "compat.md", "compat keyword", chunk_id="c-compat")
    FakeVectorStore.points.append(_vector_point(user.id, doc.id, child.metadata_json["chunk_id"], child.content, score=0.9, parent_id=child.metadata_json["parent_id"], filename=doc.filename))

    result = rag_service.search(user.id, "compat", top_k=3, min_score=0.1, db=db)["results"][0]
    for field in ("chunk_id", "content", "score", "source_title", "metadata"):
        assert field in result
    for field in ("retrieval_source", "vector_score", "bm25_score", "keyword_score", "final_score", "matched_terms", "query_type", "parent_context"):
        assert field in result


def test_ask_still_uses_parent_context_with_hybrid(hybrid_env):
    db, user, _ = hybrid_env
    doc, child = _add_document_with_child(db, user.id, "ask-hybrid.md", "child quote has invoice HT-77", chunk_id="c-ask")
    FakeVectorStore.points.append(_vector_point(user.id, doc.id, child.metadata_json["chunk_id"], child.content, score=0.9, parent_id=child.metadata_json["parent_id"], filename=doc.filename))

    result = rag_service.ask(user.id, "HT-77", top_k=3, min_score=0.1, db=db)
    assert "Parent for ask-hybrid.md" in result["evidence"][0]["quote"]
    assert "child quote" in result["evidence"][0]["child_quote"]


def test_ask_document_summary_still_prefers_overview(hybrid_env):
    db, user, _ = hybrid_env
    doc = Document(
        user_id=user.id,
        filename="overview.md",
        file_path="storage/uploads/fake/overview.md",
        file_type="md",
        source_type="user_upload",
        status="ingested",
        metadata_json={"overview": {"summary_text": "Overview remains first."}, "document_map": {"sections": []}},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    _add_child(db, user.id, doc.id, "child summary keyword", chunk_id="c-overview", parent_id="p-overview")

    result = rag_service.ask_document(user.id, "summary", document_ids=[doc.id], top_k=3, db=db)
    assert result["evidence"][0]["chunk_id"] == "overview"
    assert "Overview remains first" in result["evidence"][0]["quote"]


def test_qdrant_unavailable_returns_bm25_fallback(hybrid_env, monkeypatch):
    db, user, _ = hybrid_env
    _add_document_with_child(db, user.id, "fallback.md", "BM25 fallback keyword HT-FALLBACK", chunk_id="c-fallback")
    import src.web_app.services.rag_service as rag_service_module
    monkeypatch.setattr(rag_service_module.settings, "qdrant_url", "")

    result = rag_service.search(user.id, "HT-FALLBACK", top_k=3, min_score=0.1, db=db)
    assert result["results"]
    assert result["results"][0]["retrieval_source"] == "bm25"
    assert "retrieval_warning" in result


def test_bm25_failure_falls_back_to_vector(hybrid_env, monkeypatch):
    db, user, _ = hybrid_env
    doc, child = _add_document_with_child(db, user.id, "vector-only.md", "vector survives", chunk_id="c-vector-only")
    FakeVectorStore.points.append(_vector_point(user.id, doc.id, child.metadata_json["chunk_id"], child.content, score=0.9, parent_id=child.metadata_json["parent_id"], filename=doc.filename))
    import src.web_app.rag.retriever as retriever_module
    monkeypatch.setattr(retriever_module, "bm25_search", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bm25 boom")))

    result = rag_service.search(user.id, "vector survives", top_k=3, min_score=0.1, db=db)["results"][0]
    assert result["retrieval_source"] == "vector"
    assert result["vector_score"] > 0


def _add_document_with_child(db, user_id: int, filename: str, content: str, *, chunk_id: str, parent_id: str = "p-1", metadata: dict | None = None):
    document = Document(user_id=user_id, filename=filename, file_path=f"storage/uploads/fake/{filename}", file_type=filename.rsplit(".", 1)[-1], source_type="user_upload", status="ingested", metadata_json={})
    db.add(document)
    db.commit()
    db.refresh(document)
    parent = DocumentChunk(user_id=user_id, document_id=document.id, chunk_index=0, content=f"# Parent for {filename}\n\n{content}", token_count=10, qdrant_point_id="", metadata_json={"chunk_role": "parent", "chunk_id": parent_id})
    db.add(parent)
    db.commit()
    child = _add_child(db, user_id, document.id, content, chunk_id=chunk_id, parent_id=parent_id, metadata=metadata)
    return document, child


def _add_child(db, user_id: int, document_id: int, content: str, *, chunk_id: str, parent_id: str, metadata: dict | None = None):
    child_metadata = {"chunk_role": "child", "chunk_id": chunk_id, "parent_id": parent_id, "chunk_type": "section", "heading_path": []}
    child_metadata.update(metadata or {})
    chunk = DocumentChunk(user_id=user_id, document_id=document_id, chunk_index=10 + len(chunk_id), content=content, token_count=max(1, len(content) // 4), qdrant_point_id=f"point-{chunk_id}", metadata_json=child_metadata)
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def _vector_point(user_id: int, document_id: int, chunk_id: str, content: str, *, score: float, parent_id: str, filename: str):
    return {
        "user_id": str(user_id),
        "document_id": str(document_id),
        "chunk_id": chunk_id,
        "qdrant_point_id": f"point-{chunk_id}",
        "chunk_index": 1,
        "content": content,
        "content_preview": content[:200],
        "source_title": filename,
        "source_url": None,
        "filename": filename,
        "file_type": filename.rsplit(".", 1)[-1],
        "metadata": {"chunk_role": "child", "chunk_id": chunk_id, "parent_id": parent_id},
        "parent_id": parent_id,
        "score": score,
    }
