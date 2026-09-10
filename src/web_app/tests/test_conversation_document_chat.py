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
from src.web_app.agent.runtime.chat_fast_path import RouteHeader, RouteProtocolError, chat_entry
from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_chat_fast_path import initial, fake_model


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
async def test_followup_reads_document_without_heavy_graph(env, monkeypatch):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    calls, closed, _ = fake_model(monkeypatch, [json.dumps({"action": "document", "document_ids": [did], "mode": "overview", "query": "总结文件"}) + "\n"])
    answers = []
    async def answer(prompt):
        assert "抓包并分析" in prompt[1].content
        answers.append(prompt)
        yield SimpleNamespace(content="这是网络协议实验报告。")
    monkeypatch.setattr("src.web_app.agent.runtime.document_chat.get_chat_model", lambda *a, **kw: SimpleNamespace(astream=answer))
    from src.web_app.agent.runtime.graph import AgentRuntime
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    with env.factory() as db:
        runtime = AgentRuntime(db, {})
        async def forbidden(state):
            pytest.fail("document entered heavy workflow")
        for name in ("home_intent_react", "planner", "parallel_prefetch", "final_response"):
            monkeypatch.setattr(runtime.nodes, name, forbidden)
        state = await runtime.run({**initial(env), "user_input": "刚刚的文件是什么内容？"})
        assert state["final_answer"] == "这是网络协议实验报告。"
        assert state["file_context"]["document_ids"] == [did]
        assert len(calls) == len(answers) == 1
        assert "实验报告.docx" in calls[0][1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["chat", "clarify"])
async def test_topic_change_or_clarify_does_not_read(env, monkeypatch, action):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    fake_model(monkeypatch, [json.dumps({"action": action, "document_ids": [did]}) + "\n请说明。"])
    monkeypatch.setattr("src.web_app.agent.runtime.document_chat.read_documents", lambda *a: pytest.fail("unexpected read"))
    from src.web_app.agent.runtime.nodes import RuntimeNodes
    with env.factory() as db:
        result = await chat_entry(RuntimeNodes(db, {}), initial(env))
        assert result["final_answer"] == "请说明。"
        if action == "clarify":
            assert result["file_context"]["document_ids"] == [did]


@pytest.mark.asyncio
async def test_forged_document_scope_falls_back_without_read(env, monkeypatch):
    add_file(env)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    fake_model(monkeypatch, ['{"action":"document","document_ids":[999],"mode":"overview","query":"x"}\n'])
    from src.web_app.agent.runtime.nodes import RuntimeNodes
    with env.factory() as db:
        result = await chat_entry(RuntimeNodes(db, {}), initial(env))
        assert "你想查看哪份文件" in result["final_answer"]
        assert 999 not in result["file_context"]["document_ids"]


@pytest.mark.parametrize("split", range(1, 20))
def test_structured_header_split(split):
    value = '{"action":"clarify","document_ids":[1,2]}\n哪份？'
    parser = RouteHeader()
    assert parser.feed(value[:split]) + parser.feed(value[split:]) == "哪份？"
    assert parser.route == "clarify"


def test_complete_json_at_stream_end_still_requires_scope_validation():
    from src.web_app.agent.runtime.chat_fast_path import validate_file_decision
    parser = RouteHeader()
    parser.feed('{"action":"document","document_ids":[999],"mode":"overview","query":"总结"}')
    parser.finish()
    with pytest.raises(RouteProtocolError):
        validate_file_decision(parser, [{"document_id": 1}])


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
    did = add_file(env)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    fake_model(monkeypatch, [json.dumps({"action": "document", "document_ids": [did], "mode": "overview", "query": "总结"}) + "\n"])
    entered, release = threading.Event(), threading.Event()
    def slow_read(*args):
        entered.set()
        release.wait(3)
        return []
    monkeypatch.setattr("src.web_app.agent.runtime.document_chat.read_documents", slow_read)
    from src.web_app.agent.runtime.nodes import RuntimeNodes
    with env.factory() as db:
        task = asyncio.create_task(chat_entry(RuntimeNodes(db, {}), initial(env)))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            start = perf_counter()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert perf_counter() - start < 2
        finally:
            release.set()
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
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    monkeypatch.setattr("src.web_app.services.summary_tasks.summary_tasks.schedule", lambda *a: None)
    ctx = ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "fake", "Fake")
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_run_model_context", lambda *a, **kw: ctx)
    fake_model(monkeypatch, [json.dumps({"action": action, "document_ids": [did], "mode": "overview", "query": "总结文件"}) + "\n你指的是实验报告吗？"])
    async def answer(prompt):
        assert "抓包并分析" in prompt[1].content
        yield SimpleNamespace(content="报告介绍网络协议实验。")
    monkeypatch.setattr("src.web_app.agent.runtime.document_chat.get_chat_model", lambda *a, **kw: SimpleNamespace(astream=answer))
    with env.factory() as db:
        await execute_prepared_run(db, env.user, env.run, {"attachment_ids": [did]} if action == "document" else {})
    with env.factory() as db:
        assert file_discussion(db, env.user, "chat")["document_ids"] == [did]
        assert get_conversation(db, env.user, "chat")["files"][0]["document_id"] == did
        if action == "document":
            assert file_discussion(db, env.user, "chat")["reads"][0]["coverage"] == "full"


@pytest.mark.asyncio
async def test_document_flag_off_keeps_validated_legacy_scope(env, monkeypatch):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", False)
    fake_model(monkeypatch, [json.dumps({"action": "document", "document_ids": [did], "mode": "overview", "query": "总结"}) + "\n"])
    from src.web_app.agent.runtime.nodes import RuntimeNodes
    with env.factory() as db:
        nodes = RuntimeNodes(db, {})
        state = await chat_entry(nodes, initial(env))
        assert state["chat_entry_route"] == "workflow"
        assert nodes.payload["attachment_ids"] == [did]


@pytest.mark.asyncio
async def test_mixed_image_attachment_keeps_legacy_entry(env, monkeypatch):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    calls, _, _ = fake_model(monkeypatch, ["chat\nwrong"])
    from src.web_app.agent.runtime.nodes import RuntimeNodes
    with env.factory() as db:
        state = await chat_entry(RuntimeNodes(db, {"attachment_ids": [did, 999], "attachment_context": "existing image understanding"}), initial(env))
        assert state["chat_entry_route"] == "workflow" and calls == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("DOCUMENT_REAL_SMOKE") != "1", reason="opt-in isolated provider smoke")
async def test_real_document_followup_smoke(env, monkeypatch):
    from src.web_app.db.session import SessionLocal
    from src.web_app.models.orm import AgentRun
    from src.web_app.services.llm_registry_service import resolve_run_model_context
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.agent.runtime.nodes import RuntimeNodes
    with SessionLocal() as source:
        run = source.scalar(select(AgentRun).where(AgentRun.status == "completed").order_by(AgentRun.id.desc()))
        if not run:
            pytest.skip("No configured model")
        ctx = resolve_run_model_context(source, run.user_id, (run.graph_state or {}).get("model_context") or {})
    did = add_file(env)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    with use_model_context(ctx), env.factory() as db:
        started = perf_counter()
        result = await chat_entry(RuntimeNodes(db, {}), {**initial(env), "user_input": "刚刚上传的实验报告讲什么？"})
        assert result["file_context"]["document_ids"] == [did]
        events = db.scalars(select(AgentEvent).order_by(AgentEvent.id)).all()
        assert not any(e.event_type == "chat_route_fallback" for e in events), [e.payload_json for e in events if e.event_type == "chat_route_fallback"]
        assert "协议" in result["final_answer"] or "抓包" in result["final_answer"]
        print("DOCUMENT_REAL_SMOKE " + json.dumps({"model": ctx.model, "elapsed_ms": round((perf_counter()-started)*1000),
            "answer_chars": len(result["final_answer"]), "stages": [e.payload_json for e in events if e.event_type == "chat_latency"]}))


@pytest.mark.asyncio
async def test_document_fake_latency_samples(env, monkeypatch):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    fake_model(monkeypatch, [json.dumps({"action": "document", "document_ids": [did], "mode": "overview", "query": "总结"}) + "\n"])
    async def answer(prompt):
        yield SimpleNamespace(content="报告介绍网络协议实验。")
    monkeypatch.setattr("src.web_app.agent.runtime.document_chat.get_chat_model", lambda *a, **kw: SimpleNamespace(astream=answer))
    from src.web_app.agent.runtime.nodes import RuntimeNodes
    durations = []
    with env.factory() as db:
        for _ in range(30):
            start = perf_counter()
            await chat_entry(RuntimeNodes(db, {}), initial(env))
            durations.append((perf_counter() - start) * 1000)
    print("DOCUMENT_FAKE_LATENCY " + json.dumps({"samples": 30, "entry_total_p95_ms": round(sorted(durations)[28], 1)}))
    assert sorted(durations)[28] <= 500
