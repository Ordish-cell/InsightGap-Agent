"""Uploaded attachments carry model selection and cannot end at filename-only chat."""
import pytest
import os
from types import SimpleNamespace
from sqlalchemy import select
from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_conversation_document_chat import add_file


def test_chat_attachment_readiness_does_not_wait_for_summary_model(env, monkeypatch):
    from src.web_app.services.document_service import document_service
    from src.web_app.models.orm import Document
    from src.web_app.services.document_chat_reader import read_documents_in_session
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_model_context", lambda *a, **k: pytest.fail("Chat upload waited for model"))
    text = "Source content " * 500
    did = add_file(env, text=text)
    with env.factory() as db:
        doc = db.get(Document, did)
        doc.metadata_json = {**doc.metadata_json, "upload_type": "chat"}
        result = document_service._build_summary(db, env.user, doc, [{"content": text}])
        assert result["status"] == "skipped" and not result["summary_text"]
        db.commit()
        rows = read_documents_in_session(db, env.user, "chat", {"document_ids": [did], "mode": "overview"})
        assert rows[0]["text"] == text
        assert rows[0]["coverage"] == "full"


def test_summary_extracts_only_answer_blocks(monkeypatch):
    from src.web_app.rag.document_summarizer import _llm_summary
    message = SimpleNamespace(content=[
        {"type": "reasoning", "summary": [{"text": "not source evidence"}]},
        {"type": "text", "text": "Actual document summary"},
    ])
    monkeypatch.setattr("src.web_app.agent.llm.factory.get_chat_model", lambda *a, **k: SimpleNamespace(invoke=lambda prompt: message))
    assert _llm_summary("summarize") == "Actual document summary"
    message.content = [{"type": "reasoning", "summary": []}]
    with pytest.raises(RuntimeError, match="empty content"):
        _llm_summary("summarize")

@pytest.mark.asyncio
async def test_document_read_keeps_budget_with_long_history(env, monkeypatch):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.tests.test_native_supervisor import Model, call
    from src.web_app.tests.test_chat_fast_path import run_native, initial
    from langchain_core.messages import AIMessageChunk
    did = add_file(env, text="Unique document body")
    model = Model([[call("document.read", {"document_ids": [did]})], [AIMessageChunk(content="Document answer")]])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    with env.factory() as db:
        result = await run_native(SupervisorNodes(db, {}), {**initial(env), "context": {"conversation_history": "Old chat " * 5000}})
    assert "Unique document body" in str(model.requests[-1])
    import tiktoken
    assert len(tiktoken.get_encoding("cl100k_base").encode(str(model.requests[-1]))) < 16000
    assert result["final_answer"] == "Document answer"

@pytest.mark.asyncio
async def test_upload_persists_selected_model_before_background_start(env, monkeypatch):
    from io import BytesIO
    from starlette.datastructures import UploadFile
    from src.web_app.api.v1.documents import chat_upload
    from src.web_app.services.document_service import document_service
    from src.web_app.services.document_ingest_task_manager import document_ingest_task_manager
    from src.web_app.tests.db_test_utils import configure_test_model
    from src.web_app.services.llm_registry_service import resolve_model_context
    from src.web_app.models.orm import Document, User
    with env.factory() as db:
        configure_test_model(db, db.get(User, env.user))
        model_id = resolve_model_context(db, env.user, None).model_config_id
        doc = Document(user_id=env.user, filename="upload.md", file_path="upload.md", status="processing", metadata_json={})
        db.add(doc)
        db.commit()
        monkeypatch.setattr(document_service, "upload_chat_attachment", lambda *a: {"document_id": doc.id, "kind": "document", "status": "processing"})
        started = []
        def start(user, document):
            with env.factory() as other:
                started.append(other.get(Document, document).metadata_json["summary_model_config_id"])
        monkeypatch.setattr(document_ingest_task_manager, "start", start)
        response = await chat_upload(UploadFile(BytesIO(b"body"), filename="upload.md"), user_id=env.user, db=db, model_config_id=model_id)
    assert response["success"]
    assert started == [model_id]


@pytest.mark.asyncio
async def test_current_attachment_is_read_before_answer(env, monkeypatch):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.tests.test_native_supervisor import Model, call
    from src.web_app.tests.test_chat_fast_path import run_native, initial
    from langchain_core.messages import AIMessageChunk
    older = add_file(env, name="older.md", text="Older source")
    current = add_file(env, name="current.md", text="Current unique content")
    model = Model([[call("document.read", {"document_ids": [current]})], [AIMessageChunk(content="Current unique content [E1]")]])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    with env.factory() as db:
        result = await run_native(SupervisorNodes(db, {"attachment_ids": [current]}), initial(env))
    assert result["file_context"]["document_ids"] == [current]
    assert "Current unique content" in str(model.requests[-1])
    assert result["observations"][0]["evidence"][0]["document_id"] == current


def test_background_summary_resolves_document_owner_model(env, monkeypatch):
    from src.web_app.tests.db_test_utils import configure_test_model
    from src.web_app.models.orm import User, Document
    from src.web_app.services.document_service import document_service
    from src.web_app.agent.llm.context import get_model_context
    from src.web_app.agent.llm.errors import LLMUnavailableError
    seen = []
    def model(*a, **k):
        seen.append(get_model_context().model_config_id)
        return SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="已概括文档内容"))
    monkeypatch.setattr("src.web_app.agent.llm.factory.get_chat_model", model)
    with env.factory() as db:
        configure_test_model(db, db.get(User, env.user))
        doc = Document(user_id=env.user, filename="daily.md", metadata_json={})
        result = document_service._build_summary(db, env.user, doc, [{"content": "正文内容。" * 1000, "metadata": {}}])
    assert result["status"] == "generated"
    assert result["method"] == "hierarchical_llm_v1"
    assert len(seen) >= 2 and len(set(seen)) == 1
    with pytest.raises(LLMUnavailableError):
        get_model_context()


def test_summary_never_uses_another_users_model(env, monkeypatch):
    from src.web_app.tests.db_test_utils import configure_test_model
    from src.web_app.models.orm import User, Document
    from src.web_app.services.document_service import document_service
    from src.web_app.services.llm_registry_service import resolve_model_context
    monkeypatch.setattr("src.web_app.agent.llm.factory.get_chat_model", lambda *a, **k: pytest.fail("Foreign model executed"))
    with env.factory() as db:
        configure_test_model(db, db.get(User, env.user))
        model_id = resolve_model_context(db, env.user, None).model_config_id
        other = User(email="summary-other@example.test", hashed_password="x")
        db.add(other)
        db.commit()
        doc = Document(user_id=other.id, filename="daily.md", metadata_json={"summary_model_config_id": model_id})
        result = document_service._build_summary(db, other.id, doc, [{"content": "content " * 1000}])
    assert result["status"] == "failed"
    assert result["error"] == "model_not_found"


@pytest.mark.asyncio
@pytest.mark.skipif(os.environ.get("ATTACHMENT_LIVE_SMOKE") != "1", reason="Opt-in selected model and copied user document; isolated SQL reader")
async def test_real_model_reads_uploaded_document_and_builds_summary(env, monkeypatch):
    import asyncio
    from src.web_app.db.session import SessionLocal
    from src.web_app.models.orm import AgentRun, Document, DocumentChunk
    from src.web_app.services.llm_registry_service import resolve_run_model_context
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.services.document_service import document_service
    from src.web_app.services.document_chat_reader import read_documents_in_session
    from src.web_app.services.rag_service import rag_service
    from src.web_app.services.memory_service import memory_service
    from src.web_app.agent.runtime.graph import AgentRuntime
    from src.web_app.tests.test_supervisor_loop import state
    from src.web_app.core.config import settings
    with SessionLocal() as source:
        run = source.get(AgentRun, int(os.environ["ATTACHMENT_SOURCE_RUN"]))
        original = source.get(Document, int(os.environ["ATTACHMENT_SOURCE_DOCUMENT"]))
        assert run and original and run.user_id == original.user_id
        context = resolve_run_model_context(source, run.user_id, run.graph_state["model_context"])
        chunks = [{"content": c.content, "metadata": c.metadata_json or {}} for c in source.scalars(select(DocumentChunk).where(DocumentChunk.document_id == original.id, DocumentChunk.user_id == original.user_id))]
        parents = [c for c in chunks if c["metadata"].get("chunk_role") == "parent"] or chunks
        filename = original.filename
    def resolve(db, user, model, **kw):
        assert user == env.user
        return context
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_model_context", resolve)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    monkeypatch.setattr(memory_service, "get_baseline_memories", lambda *a, **k: [])
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", False)
    current = add_file(env, name=filename, text="\n\n".join(c["content"] for c in parents))
    with env.factory() as db:
        doc = db.get(Document, current)
        summary = await asyncio.to_thread(document_service._build_summary, db, env.user, doc, parents)
        assert summary["status"] == "generated", summary.get("error")
        assert summary["method"] == "hierarchical_llm_v1"
        doc.metadata_json = {**doc.metadata_json, "overview": {"summary_text": summary["summary_text"], "summary_status": summary["status"]}}
        db.commit()
    # Workflow retrieval also reads the copied SQL source; never query the development vector collection.
    reads = []
    def retrieve(user, question, *, db, document_ids=None, **kw):
        assert user == env.user and document_ids == [current]
        rows = read_documents_in_session(db, user, "chat", {"document_ids": document_ids, "mode": "overview", "query": question})
        reads.extend(rows)
        return {"retrieval_status": "ok", "answer": rows[0]["text"], "evidence": [{"evidence_id": "E1", "quote": rows[0]["text"], "source_title": filename, "document_id": current}]}
    monkeypatch.setattr(rag_service, "ask", retrieve)
    monkeypatch.setattr(rag_service, "ask_document", retrieve)
    monkeypatch.setattr(rag_service, "search", lambda *a, **k: pytest.fail("Development vector search is forbidden in this smoke test"))
    with use_model_context(context), env.factory() as db:
        result = await AgentRuntime(db, {"attachment_ids": [current]}).run({**state(env), "user_input": "这个呢？", "model_context": context.public_dict()})
    assert result["status"] == "completed", result.get("errors")
    assert result.get("file_context") or reads, "Answered without reading the file"
    assert result.get("final_answer")
    print("ATTACHMENT_LIVE_RESULT", {"model": context.model, "summary_method": summary["method"], "summary_chars": len(summary["summary_text"]), "document_read": True, "status": result["status"]})
