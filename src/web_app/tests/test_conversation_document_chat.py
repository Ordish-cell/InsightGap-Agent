import asyncio
import json
import os
from time import perf_counter
from types import SimpleNamespace
import pytest
from sqlalchemy import select
from src.web_app.models.orm import Document, DocumentChunk, AgentChatMessage, AgentEvent
from src.web_app.services.conversation_files import conversation_files
from src.web_app.services.document_chat_reader import read_documents_in_session
from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_chat_fast_path import initial, fake_model, run_native


def add_file(env, name="实验报告.docx", text="实验目的：验证网络协议。实验步骤：抓包并分析。", conversation="chat", status="ingested"):
    with env.factory() as db:
        doc = Document(user_id=env.user, filename=name, file_path="unused", file_type="docx", status=status,
                       metadata_json={"kind": "document", "ingest_status": status})
        db.add(doc)
        db.flush()
        db.add(DocumentChunk(user_id=env.user, document_id=doc.id, chunk_index=0, content=text))
        db.add(AgentChatMessage(message_id=f"file-{doc.id}", user_id=env.user, conversation_id=conversation,
            role="user", content="查看附件", metadata_json={"attachments": [{"document_id": doc.id, "filename": name, "kind": "document"}]}))
        db.commit()
        return doc.id


def test_inventory_survives_long_history_and_reloads_status(env):
    did = add_file(env)
    with env.factory() as db:
        for i in range(205):
            db.add(AgentChatMessage(message_id=f"later-{i}", user_id=env.user, conversation_id="chat", role="user", content="闲聊"))
        db.commit()
        assert conversation_files(db, env.user, "chat")[0]["document_id"] == did
        assert conversation_files(db, env.user, "other") == []
        db.get(Document, did).status = "failed"
        db.get(Document, did).metadata_json = {"ingest_status": "failed"}
        db.commit()
        assert conversation_files(db, env.user, "chat")[0]["status"] == "failed"


def test_scoped_reader_full_partial_and_other_conversation(env):
    did = add_file(env)
    other = add_file(env, conversation="other")
    with env.factory() as db:
        decision = {"document_ids": [did], "mode": "overview", "query": "总结"}
        assert read_documents_in_session(db, env.user, "chat", decision)[0]["coverage"] == "full"
        partial = read_documents_in_session(db, env.user, "chat", decision, 12)[0]
        assert partial["coverage"] == "partial" and len(partial["text"].encode()) <= 12
        with pytest.raises(ValueError):
            read_documents_in_session(db, env.user, "chat", {**decision, "document_ids": [other]})


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_version", [2])
async def test_followup_reads_document_without_heavy_graph(env, monkeypatch, runtime_version):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    from src.web_app.tests.test_native_supervisor import Model, call
    from langchain_core.messages import AIMessageChunk
    model = Model([[call("document.read", {"document_ids": [did]})], [AIMessageChunk(content="Document answer")]])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    from src.web_app.agent.runtime.graph import AgentRuntime
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    with env.factory() as db:
        runtime = AgentRuntime(db, {})
        async def forbidden(state):
            pytest.fail("Direct document read entered a heavy capability")
        for name in ("capability", "tool_runtime", "deep_research"):
            monkeypatch.setattr(runtime.nodes, name, forbidden)
        result = await runtime.run(initial(env))
        assert result["final_answer"] == "Document answer"
        assert result["file_context"]["document_ids"] == [did]
        assert len(model.requests) == 2
        assert "docx" in str(model.requests[0])
        assert result["file_context"]["reads"][0]["coverage"] == "full"



@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["chat", "clarify"])
async def test_topic_change_or_clarify_does_not_read(env, monkeypatch, action):
    did = add_file(env)
    from src.web_app.tests.test_native_supervisor import Model, call
    from langchain_core.messages import AIMessageChunk
    model = Model([[call("ask_user", {"question": "Which file?", "document_ids": [did]})] if action == "clarify" else [AIMessageChunk(content="New topic")]])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.read_documents_in_session", lambda *a, **k: pytest.fail("Unexpected read"))
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    with env.factory() as db:
        result = await run_native(SupervisorNodes(db, {}), initial(env))
        assert result["status"] == "completed"
        assert not result.get("observations")
        if action == "clarify":
            assert result["file_context"]["document_ids"] == [did]


@pytest.mark.asyncio
async def test_forged_document_scope_falls_back_without_read(env, monkeypatch):
    did = add_file(env)
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.tests.test_native_supervisor import Model, call
    from langchain_core.messages import AIMessageChunk
    model = Model([[call("document.read", {"document_ids": [999]})], [AIMessageChunk(content="File unavailable")]])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    with env.factory() as db:
        result = await run_native(SupervisorNodes(db, {"attachment_ids": [did]}), initial(env))
    assert result["observations"][0]["status"] == "failed"
    assert not result.get("file_context")






@pytest.mark.parametrize("status", ["processing", "failed", "uploaded"])
def test_reader_preserves_actual_unready_status(env, status):
    did = add_file(env, status=status)
    with env.factory() as db:
        result = read_documents_in_session(db, env.user, "chat", {"document_ids": [did], "mode": "overview"})[0]
        assert result["status"] == status and not result["text"]


def test_comparison_budget_is_shared_and_scope_is_per_document(env):
    ids = [add_file(env, name=f"report-{i}.docx", text="a" * 1000) for i in range(2)]
    with env.factory() as db:
        results = read_documents_in_session(db, env.user, "chat", {"document_ids": ids, "mode": "overview"}, 100)
        assert [len(r["text"]) for r in results] == [50, 50]
        assert all(r["coverage"] == "partial" for r in results)
        assert [r["references"][0]["document_id"] for r in results] == ids


def test_short_document_search_needs_no_qdrant(env, monkeypatch):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.services.rag_service.rag_service.search", lambda *a, **kw: pytest.fail("short text should use local read"))
    with env.factory() as db:
        assert read_documents_in_session(db, env.user, "chat", {"document_ids": [did], "mode": "search", "query": "步骤"})[0]["coverage"] == "full"


def test_long_document_search_failure_stays_scoped(env, monkeypatch):
    did = add_file(env, text="a" * 1000)
    def fail(*args, **kwargs):
        assert kwargs["document_ids"] == [did]
        raise RuntimeError("offline")
    monkeypatch.setattr("src.web_app.services.rag_service.rag_service.search", fail)
    with env.factory() as db:
        result = read_documents_in_session(db, env.user, "chat", {"document_ids": [did], "mode": "search", "query": "步骤"}, 100)[0]
        assert result["status"] == "retrieval_failed" and not result["text"]


@pytest.mark.asyncio
async def test_document_read_cancellation_discards_late_thread_result(env, monkeypatch):
    import threading
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.runtime.chat_control import controlled_node
    did = add_file(env)
    entered, release = threading.Event(), threading.Event()
    def slow_read(*args, **kwargs):
        entered.set()
        release.wait(3)
        return []
    monkeypatch.setattr("src.web_app.services.document_chat_reader.read_documents_in_session", slow_read)
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = initial(env)
        await nodes.permission_guard(s)
        s["current_action"] = {"action": "document_read", "action_id": "read", "arguments": {"document_ids": [did]}}
        task = asyncio.create_task(controlled_node("document_read", nodes.document_read, db)(s))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            started = perf_counter()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert perf_counter() - started < 2
        finally:
            release.set()
        assert not s.get("observations")
        events = db.scalars(select(AgentEvent)).all()
        assert any(e.event_type == "node_cancelled" for e in events)
        assert not any(e.event_type == "answer_delta" for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clarify", "document"])
async def test_completed_file_context_survives_persistence(env, monkeypatch, action):
    from src.web_app.agent.llm.context import ModelExecutionContext
    from src.web_app.services.agent_service import execute_prepared_run, get_conversation
    from src.web_app.services.conversation_files import file_discussion
    did = add_file(env)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    monkeypatch.setattr("src.web_app.services.summary_tasks.summary_tasks.schedule", lambda *a: None)
    ctx = ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "fake", "Fake")
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_run_model_context", lambda *a, **kw: ctx)
    from src.web_app.tests.test_native_supervisor import Model, call
    from langchain_core.messages import AIMessageChunk
    turns = [[call("ask_user", {"question": "Which file?", "document_ids": [did]})]] if action == "clarify" else [
        [call("document.read", {"document_ids": [did]})], [AIMessageChunk(content="Document answer")]]
    model = Model(turns)
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    with env.factory() as db:
        await execute_prepared_run(db, env.user, env.run, {"attachment_ids": [did]} if action == "document" else {})
    with env.factory() as db:
        assert file_discussion(db, env.user, "chat")["document_ids"] == [did]
        assert get_conversation(db, env.user, "chat")["files"][0]["document_id"] == did
        if action == "document":
            assert file_discussion(db, env.user, "chat")["reads"][0]["coverage"] == "full"




@pytest.mark.asyncio
async def test_mixed_image_attachment_keeps_native_context(env, monkeypatch):
    did = add_file(env)
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    calls, _, _ = fake_model(monkeypatch, ["Image and document context"])
    with env.factory() as db:
        result = await run_native(SupervisorNodes(db, {"attachment_ids": [did], "attachment_context": "existing image understanding"}), initial(env))
    assert result["status"] == "completed"
    assert "existing image understanding" in str(calls[0])
    assert str(did) in str(calls[0])


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("DOCUMENT_REAL_SMOKE") != "1", reason="opt-in isolated provider smoke")
async def test_real_document_followup_smoke(env, monkeypatch):
    from src.web_app.db.session import SessionLocal
    from src.web_app.models.orm import AgentRun
    from src.web_app.services.llm_registry_service import resolve_run_model_context
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    with SessionLocal() as source:
        run = source.scalar(select(AgentRun).where(AgentRun.status == "completed").order_by(AgentRun.id.desc()))
        if not run:
            pytest.skip("No configured model")
        ctx = resolve_run_model_context(source, run.user_id, (run.graph_state or {}).get("model_context") or {})
    did = add_file(env)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    with use_model_context(ctx), env.factory() as db:
        started = perf_counter()
        result = await run_native(SupervisorNodes(db, {}), {**initial(env), "user_input": "刚刚上传的实验报告讲什么？"})
        assert result["file_context"]["document_ids"] == [did]
        events = db.scalars(select(AgentEvent).order_by(AgentEvent.id)).all()
        assert not any(e.event_type == "chat_route_fallback" for e in events), [e.payload_json for e in events if e.event_type == "chat_route_fallback"]
        assert "协议" in result["final_answer"] or "抓包" in result["final_answer"]
        print("DOCUMENT_REAL_SMOKE " + json.dumps({"model": ctx.model, "elapsed_ms": round((perf_counter()-started)*1000),
            "answer_chars": len(result["final_answer"]), "stages": [e.payload_json for e in events if e.event_type == "chat_latency"]}))


@pytest.mark.asyncio
async def test_document_fake_latency_samples(env, monkeypatch):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.tests.test_native_supervisor import Model, call
    from langchain_core.messages import AIMessageChunk
    did = add_file(env)
    durations = []
    for _ in range(30):
        model = Model([[call("document.read", {"document_ids": [did]})], [AIMessageChunk(content="Document answer")]])
        monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
        with env.factory() as db:
            db.query(AgentEvent).delete()
            db.commit()
            started = perf_counter()
            result = await run_native(SupervisorNodes(db, {}), initial(env))
            durations.append((perf_counter() - started) * 1000)
            assert result["file_context"]["document_ids"] == [did]
            assert len(model.requests) == 2
    assert sorted(durations)[28] <= 500
