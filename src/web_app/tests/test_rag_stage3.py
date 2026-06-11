from io import BytesIO
import hashlib

import pytest
from fastapi.testclient import TestClient

from src.web_app.api.v1.documents import get_current_user_id
from src.web_app.db.session import get_db
from src.web_app.main import app
from src.web_app.models.orm import AgentChatMessage, AgentConversation, Document, DocumentChunk, Memory, User
from src.web_app.rag import embeddings
from src.web_app.rag import document_parser
from src.web_app.rag import vector_store as vector_store_module
from src.web_app.rag.structured_chunker import build_structured_chunks
from src.web_app.services.agent_service import hard_delete_conversation
from src.web_app.services.rag_service import rag_service
from src.web_app.tests.db_test_utils import make_test_session


class FakeVectorStore:
    points: list[dict] = []

    def ensure_collection(self):
        return None

    def upsert_chunks(self, user_id, document_id, chunks, vectors, document):
        ids = []
        for index, chunk in enumerate(chunks):
            point_id = f"point-{document_id}-{index}"
            metadata = dict(chunk.get("metadata", {}))
            chunk_id = str(metadata.get("chunk_id") or point_id)
            content_hash = metadata.get("content_hash") or hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
            ids.append(point_id)
            self.points.append(
                {
                    "user_id": str(user_id),
                    "document_id": str(document_id),
                    "chunk_id": chunk_id,
                    "qdrant_point_id": point_id,
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "content_preview": chunk["content"][:200],
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "heading_path": chunk.get("heading_path", []),
                    "token_count": chunk.get("token_count", 0),
                    "content_hash": content_hash,
                    "chunk_role": metadata.get("chunk_role", "child"),
                    "chunk_type": metadata.get("chunk_type", "text"),
                    "parent_id": metadata.get("parent_id"),
                    "page_number": metadata.get("page_number"),
                    "sheet_name": metadata.get("sheet_name"),
                    "source_title": document.filename,
                    "source_url": None,
                    "metadata": metadata,
                }
            )
        return ids

    def search(self, user_id, query_vector, top_k=5, min_score=0.2, document_ids=None):
        rows = [row for row in self.points if row["user_id"] == str(user_id)]
        if document_ids:
            rows = [row for row in rows if int(row["document_id"]) in document_ids]
        return [{**row, "score": 0.9} for row in rows[:top_k]]

    def delete_document(self, user_id, document_id):
        type(self).points = [row for row in type(self).points if not (row["user_id"] == str(user_id) and row["document_id"] == str(document_id))]

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


def test_reingest_same_document_is_idempotent_for_qdrant_points(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("repeat.txt", b"# RAG\n\nsame content", "text/plain")}).json()["data"]
    first = client.post(f"/api/v1/documents/{upload['id']}/ingest").json()["data"]
    first_count = len(FakeVectorStore.points)
    second = client.post(f"/api/v1/documents/{upload['id']}/ingest").json()["data"]
    assert second["chunk_count"] == first["chunk_count"]
    assert len(FakeVectorStore.points) == first_count


def test_qdrant_payload_has_stable_chunk_id_and_content_hash(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("payload.txt", b"payload content", "text/plain")}).json()["data"]
    client.post(f"/api/v1/documents/{upload['id']}/ingest")
    point = FakeVectorStore.points[0]
    assert point["chunk_id"]
    assert point["content_hash"]


def test_structured_ingest_stores_parent_overview_and_child_only_vectors(client):
    content = b"# Intro\n\nAlpha evidence.\n\n## Details\n\nBeta evidence for retrieval."
    upload = client.post("/api/v1/documents/upload", files={"file": ("structured.md", content, "text/markdown")}).json()["data"]
    ingest = client.post(f"/api/v1/documents/{upload['id']}/ingest")
    assert ingest.status_code == 200

    db = next(app.dependency_overrides[get_db]())
    document = db.get(Document, upload["id"])
    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == upload["id"]).order_by(DocumentChunk.chunk_index).all()
    roles = [chunk.metadata_json.get("chunk_role") for chunk in chunks]
    child_chunks = [chunk for chunk in chunks if chunk.metadata_json.get("chunk_role") == "child"]

    assert "overview" in roles
    assert "parent" in roles
    assert child_chunks
    assert all(chunk.metadata_json.get("parent_id") for chunk in child_chunks)
    assert all(point["chunk_role"] == "child" for point in FakeVectorStore.points)
    assert len(FakeVectorStore.points) == len(child_chunks)
    assert document.metadata_json["chunk_count"] == len(child_chunks)
    assert document.metadata_json["chunk_count"] < len(chunks)
    assert document.metadata_json["overview"]["summary_text"]
    assert document.metadata_json["document_map"]["sections"]


def test_structured_csv_ingest_uses_row_blocks(client):
    rows = ["name,score,comment"] + [f"user{i},{i},note {i}" for i in range(45)]
    upload = client.post("/api/v1/documents/upload", files={"file": ("scores.csv", "\n".join(rows).encode("utf-8"), "text/csv")}).json()["data"]
    ingest = client.post(f"/api/v1/documents/{upload['id']}/ingest")
    assert ingest.status_code == 200

    db = next(app.dependency_overrides[get_db]())
    document = db.get(Document, upload["id"])
    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == upload["id"]).order_by(DocumentChunk.chunk_index).all()
    child_chunks = [chunk for chunk in chunks if chunk.metadata_json.get("chunk_role") == "child"]

    assert child_chunks
    assert all(chunk.metadata_json.get("chunk_type") == "row_block" for chunk in child_chunks)
    assert all(chunk.metadata_json.get("header") == ["name", "score", "comment"] for chunk in child_chunks)
    assert all("Columns: name | score | comment" in chunk.content for chunk in child_chunks)
    assert document.metadata_json["chunk_count"] == len(child_chunks)
    assert len(FakeVectorStore.points) == len(child_chunks)


def test_structured_xlsx_text_uses_sheet_row_blocks():
    parsed_text = "# Budget\nteam\tamount\ninfra\t120\nresearch\t240\n"
    result = build_structured_chunks(parsed_text, file_type="xlsx", filename="budget.xlsx", parser_metadata={"parser": "openpyxl", "sheet_names": ["Budget"]})
    child_chunks = [chunk for chunk in result["chunks"] if chunk["metadata"].get("chunk_role") == "child"]

    assert child_chunks
    assert all(chunk["metadata"].get("chunk_type") == "row_block" for chunk in child_chunks)
    assert child_chunks[0]["metadata"]["sheet_name"] == "Budget"
    assert child_chunks[0]["metadata"]["header"] == ["team", "amount"]
    assert result["stats"]["chunk_count"] == len(result["vector_chunks"]) == len(child_chunks)


def test_structured_chunker_failure_falls_back_to_legacy(client, monkeypatch):
    import src.web_app.services.document_service as document_service_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("structured chunk failed")

    monkeypatch.setattr(document_service_module, "build_structured_chunks", boom)
    upload = client.post("/api/v1/documents/upload", files={"file": ("fallback-chunk.txt", b"plain fallback content", "text/plain")}).json()["data"]
    response = client.post(f"/api/v1/documents/{upload['id']}/ingest")
    assert response.status_code == 200

    db = next(app.dependency_overrides[get_db]())
    document = db.get(Document, upload["id"])
    assert document.metadata_json["chunking_stats"]["chunking_strategy"] == "legacy_chunk_markdown"
    assert "structured chunk failed" in document.metadata_json["chunking_stats"]["structured_error"]


def test_rag_search_returns_evidence(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("search.txt", b"Agent OS uses evidence.", "text/plain")}).json()["data"]
    client.post(f"/api/v1/documents/{upload['id']}/ingest")
    response = client.post("/api/v1/rag/search", json={"query": "evidence", "top_k": 3, "min_score": 0.1})
    assert response.json()["data"]["results"]


def test_parent_child_search_enriches_child_hit_with_parent_context(client):
    content = b"# Parent Section\n\nParent context explains retrieval.\n\nChild detail mentions needle."
    upload = client.post("/api/v1/documents/upload", files={"file": ("parent-search.md", content, "text/markdown")}).json()["data"]
    client.post(f"/api/v1/documents/{upload['id']}/ingest")

    response = client.post("/api/v1/rag/search", json={"query": "needle", "top_k": 3, "min_score": 0.1})
    result = response.json()["data"]["results"][0]

    assert result["chunk_id"]
    assert result["content"]
    assert result["score"] == 0.9
    assert result["source_title"] == "parent-search.md"
    assert result["child_chunk_id"] == result["chunk_id"]
    assert result["parent_id"]
    assert result["parent_context_available"] is True
    assert "Parent Section" in result["parent_context"]


def test_parent_child_search_falls_back_when_parent_missing(client):
    db = next(app.dependency_overrides[get_db]())
    document = Document(user_id=client.user_id, filename="orphan.txt", file_path="storage/uploads/fake/orphan.txt", file_type="txt", source_type="user_upload", status="ingested", metadata_json={})
    db.add(document)
    db.commit()
    db.refresh(document)
    FakeVectorStore.points.append({
        "user_id": str(client.user_id),
        "document_id": str(document.id),
        "chunk_id": "orphan-child",
        "chunk_index": 7,
        "content": "orphan child content",
        "content_preview": "orphan child content",
        "source_title": "orphan.txt",
        "source_url": None,
        "filename": "orphan.txt",
        "metadata": {"parent_id": "missing-parent"},
        "parent_id": "missing-parent",
    })

    result = rag_service.ask(client.user_id, "orphan", top_k=3, min_score=0.1, db=db)
    evidence = result["evidence"][0]
    assert evidence["quote"] == "orphan child content"
    assert evidence["child_quote"] == "orphan child content"
    assert evidence["parent_context_available"] is False


def test_ask_evidence_uses_parent_context_and_dedupes_same_parent(client):
    db = next(app.dependency_overrides[get_db]())
    document = Document(user_id=client.user_id, filename="dedupe.md", file_path="storage/uploads/fake/dedupe.md", file_type="md", source_type="user_upload", status="ingested", metadata_json={})
    db.add(document)
    db.commit()
    db.refresh(document)
    parent = DocumentChunk(
        user_id=client.user_id,
        document_id=document.id,
        chunk_index=1,
        content="# Parent\n\nfull parent context with two children",
        token_count=10,
        qdrant_point_id="",
        metadata_json={"chunk_role": "parent", "chunk_id": "p-shared"},
    )
    db.add(parent)
    db.commit()
    for idx, text in enumerate(["first child quote", "second child quote"]):
        FakeVectorStore.points.append({
            "user_id": str(client.user_id),
            "document_id": str(document.id),
            "chunk_id": f"p-shared-c-{idx}",
            "chunk_index": idx + 2,
            "content": text,
            "content_preview": text,
            "source_title": "dedupe.md",
            "source_url": None,
            "filename": "dedupe.md",
            "metadata": {"parent_id": "p-shared"},
            "parent_id": "p-shared",
        })

    result = rag_service.ask(client.user_id, "children", top_k=5, min_score=0.1, db=db)
    assert len(result["evidence"]) == 1
    assert "full parent context" in result["evidence"][0]["quote"]
    assert result["evidence"][0]["child_quote"] == "first child quote"


def test_parent_lookup_is_limited_by_user_and_document(client):
    db = next(app.dependency_overrides[get_db]())
    other_doc = Document(user_id=client.other_user_id, filename="other.md", file_path="storage/uploads/fake/other.md", file_type="md", source_type="user_upload", status="ingested", metadata_json={})
    db.add(other_doc)
    db.commit()
    db.refresh(other_doc)
    db.add(DocumentChunk(
        user_id=client.other_user_id,
        document_id=other_doc.id,
        chunk_index=1,
        content="other user's parent must not leak",
        token_count=8,
        qdrant_point_id="",
        metadata_json={"chunk_role": "parent", "chunk_id": "p-leak"},
    ))
    db.commit()
    FakeVectorStore.points.append({
        "user_id": str(client.user_id),
        "document_id": str(other_doc.id),
        "chunk_id": "malicious-child",
        "chunk_index": 2,
        "content": "safe child fallback",
        "content_preview": "safe child fallback",
        "source_title": "malicious.md",
        "source_url": None,
        "filename": "malicious.md",
        "metadata": {"parent_id": "p-leak"},
        "parent_id": "p-leak",
    })

    result = rag_service.search(client.user_id, "leak", top_k=3, min_score=0.1, db=db)["results"][0]
    assert result["parent_context_available"] is False
    assert result["parent_context"] == "safe child fallback"
    assert "other user's parent" not in result["parent_context"]


def test_rag_ask_returns_answer_with_evidence(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("ask.txt", b"Evidence based answers cite chunks.", "text/plain")}).json()["data"]
    client.post(f"/api/v1/documents/{upload['id']}/ingest")
    response = client.post("/api/v1/rag/ask", json={"question": "How answer?", "top_k": 3, "min_score": 0.1})
    data = response.json()["data"]
    assert data["evidence"]
    assert data["answer_mode"] == "extractive_fallback"


def test_ask_evidence_uses_parent_context(client):
    content = b"# Evidence Parent\n\nParent context should be used for answer evidence.\n\nSpecific child fact."
    upload = client.post("/api/v1/documents/upload", files={"file": ("parent-ask.md", content, "text/markdown")}).json()["data"]
    client.post(f"/api/v1/documents/{upload['id']}/ingest")

    response = client.post("/api/v1/rag/ask", json={"question": "Specific child fact", "top_k": 3, "min_score": 0.1})
    evidence = response.json()["data"]["evidence"][0]
    assert "Evidence Parent" in evidence["quote"]
    assert "Specific child fact" in evidence["child_quote"]
    assert evidence["citation"]["child_chunk_id"] == evidence["chunk_id"]


def test_ask_document_summary_prefers_overview_and_document_map(client):
    db = next(app.dependency_overrides[get_db]())
    document = Document(
        user_id=client.user_id,
        filename="overview.md",
        file_path="storage/uploads/fake/overview.md",
        file_type="md",
        source_type="user_upload",
        status="ingested",
        metadata_json={
            "overview": {"summary_text": "Overview says this document is about parent-child retrieval."},
            "document_map": {"filename": "overview.md", "sections": [{"parent_id": "p-1", "heading_path": ["Intro"], "chunk_type": "section"}]},
        },
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    result = rag_service.ask_document(client.user_id, "总结这个文档", document_ids=[document.id], top_k=3, overview_mode=False, db=db)
    assert result["evidence"]
    assert result["evidence"][0]["chunk_id"] == "overview"
    assert "parent-child retrieval" in result["evidence"][0]["quote"]
    assert result["evidence"][0]["document_map"]["sections"][0]["heading_path"] == ["Intro"]


def test_rag_ask_no_evidence_does_not_hallucinate(client):
    response = client.post("/api/v1/rag/ask", json={"question": "unknown", "top_k": 3, "min_score": 0.1})
    data = response.json()["data"]
    assert data["evidence"] == []
    assert "没有找到" in data["answer"]


def test_ingested_chat_document_is_not_reingested(client):
    from src.web_app.services.document_service import document_service
    db = next(app.dependency_overrides[get_db]())
    document = Document(
        user_id=client.user_id,
        filename="chat.txt",
        file_path="storage/uploads/fake/chat.txt",
        file_type="txt",
        source_type="chat_upload",
        status="ingested",
        metadata_json={"kind": "document", "ingest_status": "ingested", "chunk_count": 2},
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    result = document_service.ingest_chat_document(db, client.user_id, document.id)
    assert result["status"] == "skipped"
    assert result["reason"] == "already_ingested"


def test_delete_document_continues_when_qdrant_not_configured(client, monkeypatch):
    import src.web_app.services.document_service as document_service_module

    upload = client.post("/api/v1/documents/upload", files={"file": ("delete-me.txt", b"delete", "text/plain")}).json()["data"]
    monkeypatch.setattr(document_service_module.settings, "qdrant_url", "")
    response = client.delete(f"/api/v1/documents/{upload['id']}")
    data = response.json()["data"]
    assert data["deleted"] is True
    assert "vector_cleanup_warning" in data
    db = next(app.dependency_overrides[get_db]())
    assert db.get(Document, upload["id"]) is None


def test_ingest_stops_when_required_qdrant_delete_fails(client, monkeypatch):
    import src.web_app.services.document_service as document_service_module

    upload = client.post("/api/v1/documents/upload", files={"file": ("qdrant-fail.txt", b"content", "text/plain")}).json()["data"]

    class FailingDeleteVectorStore(FakeVectorStore):
        def delete_document(self, user_id, document_id):
            raise RuntimeError("delete failed")

    monkeypatch.setattr(document_service_module.settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(document_service_module, "QdrantVectorStore", FailingDeleteVectorStore)
    response = client.post(f"/api/v1/documents/{upload['id']}/ingest")
    assert response.json()["success"] is False
    db = next(app.dependency_overrides[get_db]())
    document = db.get(Document, upload["id"])
    assert document.status == "failed"
    assert document.metadata_json["failed_stage"] == "qdrant_delete"


def test_markitdown_fallback_reason_is_recorded(tmp_path, monkeypatch):
    path = tmp_path / "fallback.txt"
    path.write_text("fallback text", encoding="utf-8")

    def boom(_file_path):
        raise RuntimeError("markitdown exploded")

    monkeypatch.setattr(document_parser, "_parse_markitdown", boom)
    parsed = document_parser.parse_document(path)
    assert parsed["metadata"]["used_fallback"] is True
    assert "markitdown exploded" in parsed["metadata"]["fallback_reason"]


def test_user_cannot_access_other_user_document(client):
    upload = client.post("/api/v1/documents/upload", files={"file": ("mine.txt", b"mine", "text/plain")}).json()["data"]
    app.dependency_overrides[get_current_user_id] = lambda: client.other_user_id
    response = client.get(f"/api/v1/documents/{upload['id']}")
    assert response.json()["success"] is False


def test_qdrant_search_filters_by_user_id(client):
    FakeVectorStore.points.append({"user_id": "9999", "document_id": "99", "chunk_id": "x", "chunk_index": 0, "content": "secret", "content_preview": "secret", "source_title": "x", "source_url": None, "metadata": {}})
    response = client.post("/api/v1/rag/search", json={"query": "secret", "top_k": 5, "min_score": 0.1})
    assert response.json()["data"]["results"] == []


def test_hard_delete_conversation_cleans_document_vectors_not_memory_vectors(client, monkeypatch):
    db = next(app.dependency_overrides[get_db]())
    conversation_id = "conv-rag-cleanup"
    doc = Document(
        user_id=client.user_id,
        filename="attached.txt",
        file_path="storage/uploads/fake/attached.txt",
        file_type="txt",
        source_type="chat_upload",
        status="ingested",
        metadata_json={"kind": "document", "ingest_status": "ingested"},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    conv = AgentConversation(conversation_id=conversation_id, user_id=client.user_id, status="active", metadata_json={})
    msg = AgentChatMessage(
        message_id="msg-rag-cleanup",
        conversation_id=conversation_id,
        user_id=client.user_id,
        role="user",
        content="see attachment",
        metadata_json={"attachments": [{"document_id": doc.id, "kind": "document"}]},
    )
    db.add_all([conv, msg])
    db.commit()
    FakeVectorStore.points.append({"user_id": str(client.user_id), "document_id": str(doc.id), "chunk_id": "c", "chunk_index": 0, "content": "x", "content_preview": "x", "source_title": "attached.txt", "source_url": None, "metadata": {}})

    memory_delete_calls = []

    class MemoryStoreShouldNotBeUsed:
        def delete_by_memory_id(self, memory_id):
            memory_delete_calls.append(memory_id)

    import src.web_app.services.agent_service as agent_service_module
    monkeypatch.setattr(agent_service_module, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr("src.web_app.memory.qdrant_memory_store.QdrantMemoryStore", MemoryStoreShouldNotBeUsed)

    result = hard_delete_conversation(db, client.user_id, conversation_id)
    assert result["deleted_records"] >= 1
    assert FakeVectorStore.points == []
    assert memory_delete_calls == []


def test_manual_memory_delete_removes_memory_vector(client):
    from src.web_app.services.memory_service import memory_service

    db = next(app.dependency_overrides[get_db]())
    item = Memory(user_id=client.user_id, memory_type="semantic", content="remember me", importance=0.5, source_type="", qdrant_point_id="mp")
    db.add(item)
    db.commit()
    db.refresh(item)

    deleted_ids = []

    class FakeMemoryStore:
        def delete_by_memory_id(self, memory_id):
            deleted_ids.append(str(memory_id))

    old_store = memory_service._qdrant_store
    old_attempted = memory_service._qdrant_init_attempted
    memory_service._qdrant_store = FakeMemoryStore()
    memory_service._qdrant_init_attempted = True
    try:
        result = memory_service.forget_memory(client.user_id, item.id, db)
    finally:
        memory_service._qdrant_store = old_store
        memory_service._qdrant_init_attempted = old_attempted
    assert result["deleted"] == 1
    assert deleted_ids == [str(item.id)]


def test_document_delete_does_not_call_memory_vector_cleanup(client, monkeypatch):
    calls = []

    class MemoryStoreShouldNotBeUsed:
        def delete_by_memory_id(self, memory_id):
            calls.append(memory_id)

    monkeypatch.setattr("src.web_app.memory.qdrant_memory_store.QdrantMemoryStore", MemoryStoreShouldNotBeUsed)
    upload = client.post("/api/v1/documents/upload", files={"file": ("doc-only.txt", b"doc", "text/plain")}).json()["data"]
    response = client.delete(f"/api/v1/documents/{upload['id']}")
    assert response.json()["data"]["deleted"] is True
    assert calls == []


def test_health_dependencies_includes_embedding_and_qdrant(client):
    response = client.get("/api/v1/health/dependencies")
    data = response.json()["data"]
    assert "embedding_provider" in data
    assert "qdrant_collection" in data
