"""Behavioral regression coverage for consent, provenance, scope and versions."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.web_app.models.orm import User, UserProfile, AgentConversation, AgentRun, AgentChatMessage, Memory, MemoryMaintenanceTask
from src.web_app.services.memory_service import MemoryService
from src.web_app.memory.policy import settings_for, infer_scope
from src.web_app.tests.db_test_utils import make_test_session


@pytest.fixture
def env(monkeypatch):
    db = make_test_session()
    user = User(email="memory-v2@example.test", hashed_password="x")
    db.add(user)
    db.flush()
    profile = UserProfile(user_id=user.id)
    db.add(profile)
    for cid in ("one", "two"):
        db.add(AgentConversation(user_id=user.id, conversation_id=cid))
    db.commit()
    monkeypatch.setattr(MemoryService, "_get_qdrant_store", lambda self: None)
    monkeypatch.setattr(MemoryService, "_sync_memory_graph", lambda *args: {})
    from src.web_app.memory.facts import normalized
    monkeypatch.setattr("src.web_app.services.memory_service.compare", lambda old, new, **kw:
        "duplicate" if normalized(old) == normalized(new) else "correction" if kw.get("same_key") else "ambiguous")
    yield SimpleNamespace(db=db, user=user.id, profile=profile, service=MemoryService())
    db.close()


def add(env, content, *, category="tech_stack", cid="one", **meta):
    return env.service.add_with_dedup(env.user, content, "semantic", .95,
        metadata={"category": category, "conversation_id": cid, **meta}, db=env.db)


def test_scope_personal_and_conversation(env):
    personal = add(env, "Concise answers", category="answer_preference")
    local = add(env, "Python project")
    assert personal["scope"] == "user" and local["scope_id"] == "one"
    assert {m["id"] for m in env.service.search_memory(env.user, db=env.db, conversation_id="two")} == {personal["id"]}
    assert {m["id"] for m in env.service.search_memory(env.user, db=env.db, conversation_id="one")} == {personal["id"], local["id"]}
    assert infer_scope({"category": "answer_preference", "source_quote": "本次回答简短"}, "one") == ("conversation", "one")


@pytest.mark.parametrize("metadata,expiry", [
    ({"status": "archived"}, None), ({"status": "pending"}, None),
    ({"visible_in_long_term_memory": False}, None), ({"sensitive": True}, None),
    ({}, datetime.now(UTC) - timedelta(days=1)),
])
def test_invalid_vector_hits_fall_back_without_leaking(env, monkeypatch, metadata, expiry):
    stale = add(env, "Python old fact", category="answer_preference")
    item = env.db.get(Memory, stale["id"])
    item.metadata_json = {**item.metadata_json, **metadata}
    item.expires_at = expiry
    env.db.commit()
    valid = add(env, "Python current fact", category="language_preference")
    store = SimpleNamespace(search_memory=lambda **kw: [{"memory_id": stale["id"], "score": .99}])
    monkeypatch.setattr(env.service, "_get_qdrant_store", lambda: store)
    assert [m["id"] for m in env.service.search_memory(env.user, "Python", db=env.db)] == [valid["id"]]
    assert env.service.search_memory(env.user, "unmatched-topic", db=env.db) == []


def test_correction_replaces_fact_without_merging_opposites(env):
    old = add(env, "用户喜欢使用 Python", entity="python")
    new = add(env, "用户不喜欢使用 Python", entity="python")
    assert new["content"] == "用户不喜欢使用 Python"
    assert env.db.get(Memory, old["id"]).metadata_json["status"] == "superseded"
    assert [m["id"] for m in env.service.search_memory(env.user, "Python", db=env.db, conversation_id="one")] == [new["id"]]
    assert len(env.service.search_memory(env.user, "Python", db=env.db, conversation_id="two")) == 0


def test_ambiguous_candidate_and_confirmation(env, monkeypatch):
    old = add(env, "用户喜欢 Python 开发")
    monkeypatch.setattr("src.web_app.services.memory_service.compare", lambda *args, **kw: "ambiguous")
    new = add(env, "用户不喜欢 Python 开发")
    assert new["status"] == "pending"
    assert env.db.get(Memory, old["id"]).metadata_json.get("status", "active") == "active"
    env.service.restore_memory(env.user, env.db.get(Memory, new["id"]), env.db)
    assert env.db.get(Memory, new["id"]).metadata_json["confirmed"] is True


def test_repeated_evidence_is_idempotent(env):
    first = add(env, "Python", entity="python", source_message_id=1, source_quote="Python")
    second = add(env, "Python", entity="python", source_message_id=1, source_quote="Python")
    third = add(env, "Python", entity="python", source_message_id=2, source_quote="Python")
    assert first["id"] == second["id"] == third["id"]
    assert third["metadata"]["evidence_count"] == 2


def test_unverified_or_older_evidence_cannot_replace(env):
    old = add(env, "Python", entity="stack", source_message_id=20, source_quote="Python")
    unverified = add(env, "PostgreSQL", entity="stack", status="pending")
    older = add(env, "MySQL", entity="stack", source_message_id=10, source_quote="MySQL")
    assert unverified["status"] == older["status"] == "pending"
    assert env.db.get(Memory, old["id"]).metadata_json.get("status", "active") == "active"


def test_control_precedence_and_explicit_denial(env):
    env.profile.use_memory = False
    env.profile.generate_memory = True
    conv = env.db.scalar(select(AgentConversation).where(AgentConversation.conversation_id == "one"))
    conv.metadata_json = {"memory_settings": {"use_memory": True, "generate_memory": False}}
    env.db.commit()
    assert settings_for(env.db, env.user, "one")["use_memory"] is True
    assert settings_for(env.db, env.user, "one")["generate_memory"] is False
    assert settings_for(env.db, env.user, "one", {"generate_memory": True})["generate_memory"] is True
    assert settings_for(env.db, env.user, "one", {"generate_memory": True, "write_memory": False})["generate_memory"] is False
    assert settings_for(env.db, env.user, "one", {"generate_memory": True, "user_input": "不要记住"})["writes_blocked"] is True
    assert env.service.search_memory(env.user, db=env.db, use_memory=False) == []


def setup_job(env, monkeypatch):
    from src.web_app.services import memory_tasks as module
    env.profile.generate_memory = True
    env.db.commit()
    run = AgentRun(user_id=env.user, conversation_id="one", status="completed", user_input="本项目使用 Python",
                   memory_policy=settings_for(env.db, env.user, "one"))
    env.db.add(run)
    env.db.flush()
    msg = AgentChatMessage(user_id=env.user, conversation_id="one", run_id=run.id, message_id="source",
                          role="user", content=run.user_input, status="completed")
    env.db.add(msg)
    env.db.commit()
    factory = sessionmaker(bind=env.db.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(module, "SessionLocal", factory)
    manager = module.MemoryTasks()
    monkeypatch.setattr(manager, "start", lambda *args: None)
    manager.enqueue(env.db, env.user, "one", run.id)
    return module, manager, run, env.db.scalar(select(MemoryMaintenanceTask))


def test_job_enqueue_is_unique_and_generation_default_off(env, monkeypatch):
    module, manager, run, job = setup_job(env, monkeypatch)
    manager.enqueue(env.db, env.user, "one", run.id)
    assert len(list(env.db.scalars(select(MemoryMaintenanceTask)))) == 1
    assert job.source_message_id
    from src.web_app.services.profile_service import update_profile
    update_profile(env.db, env.user, {"generate_memory": False})
    assert module.still_allowed(env.db, job) is False
    update_profile(env.db, env.user, {"generate_memory": True})
    assert module.still_allowed(env.db, job) is False  # old consent cannot be revived


def test_completed_job_validates_source_and_retry_does_not_duplicate(env, monkeypatch):
    module, manager, run, job = setup_job(env, monkeypatch)
    from src.web_app.agent.llm.context import ModelExecutionContext
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_model_context", lambda *args: ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "test", "Test"))
    async def extract(*args, **kwargs):
        return {"semantic_memories": [{"content": "本项目使用 Python", "category": "tech_stack", "entity": "stack", "importance": .95, "confidence": .95, "source_quote": "本项目使用 Python"}]}
    monkeypatch.setattr("src.web_app.memory.extractor.LlmMemoryExtractor.extract", extract)
    module.process(job.id)
    env.db.expire_all()
    assert env.db.get(MemoryMaintenanceTask, job.id).status == "completed"
    memories = list(env.db.scalars(select(Memory).where(Memory.source_type != "memory_summary")))
    assert len(memories) == 1 and memories[0].scope == "conversation"
    assert memories[0].metadata_json["source_verified"] is True
    module.process(job.id)
    assert len(list(env.db.scalars(select(Memory)))) == 2  # fact plus derived summary


def test_job_failure_retries_and_closing_cancels(env, monkeypatch):
    module, manager, run, job = setup_job(env, monkeypatch)
    def fail(*args):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_model_context", fail)
    module.process(job.id)
    env.db.expire_all()
    job = env.db.get(MemoryMaintenanceTask, job.id)
    assert job.status == "failed" and job.attempts == 1
    from src.web_app.services.agent_service import clear_conversation
    clear_conversation(env.db, env.user, "one")
    module.process(job.id)
    env.db.expire_all()
    assert env.db.get(MemoryMaintenanceTask, job.id).status == "cancelled"
    assert not list(env.db.scalars(select(Memory)))


def test_scoped_summary_retains_evidence_and_drops_stale_content(env):
    old = add(env, "Python", entity="stack")
    env.service.consolidate_memory(env.user, env.db, conversation_id="one")
    add(env, "MySQL", entity="stack")
    # The old summary is ineligible until rebuilt; stale source IDs are revalidated.
    assert not env.service.search_memory(env.user, "Python", db=env.db, conversation_id="one")
    env.service.consolidate_memory(env.user, env.db, conversation_id="one")
    assert env.db.get(Memory, old["id"]) is not None
    assert env.service.search_memory(env.user, "MySQL", db=env.db, conversation_id="one")


def test_long_message_keeps_tail_and_failed_summary_keeps_cursor(env, monkeypatch):
    from src.web_app.agent.runtime.context import format_history
    from src.web_app.services.conversation_summary_service import ConversationSummaryService, _format_messages_for_segment
    text = "背景说明 " * 8000 + "最终约束：不要修改数据库模式"
    msg = SimpleNamespace(role="user", content=text, status="completed")
    assert text in format_history([msg])
    from src.web_app.services.agent_service import _format_chat_messages_for_context
    assert text in _format_chat_messages_for_context([msg])
    assert _format_messages_for_segment([msg])[0]["content"].endswith("不要修改数据库模式")
    calls = []
    def summarize(prompt):
        calls.append(prompt)
        return '{"summary_text":"保留约束","facts":["不要修改数据库模式"]}'
    monkeypatch.setattr("src.web_app.services.conversation_summary_service._llm_call", summarize)
    service = ConversationSummaryService()
    service.update_after_turn("one", env.user, [{"role": "user", "content": text}], db=env.db, last_message_id=10)
    assert len(calls) > 1 and "不要修改数据库模式" in calls[-1]
    monkeypatch.setattr("src.web_app.services.conversation_summary_service._llm_call", lambda prompt: "invalid json")
    service.update_after_turn("one", env.user, [{"role": "user", "content": "new"}], db=env.db, last_message_id=20)
    assert service.get_summary("one", env.user, db=env.db)["last_message_id"] == 10


def test_manual_edit_is_authoritative_and_keeps_scope(env, monkeypatch):
    old = add(env, "Python", entity="stack")
    monkeypatch.setattr("src.web_app.services.memory_service.compare", lambda *a, **k: "ambiguous")
    revised = env.service.edit_memory(env.user, env.db.get(Memory, old["id"]), {"content": "PostgreSQL"}, env.db)
    assert revised["id"] != old["id"] and revised["scope_id"] == "one"
    assert revised["metadata"]["manual_evidence"]["quote"] == "PostgreSQL"
    assert env.db.get(Memory, old["id"]).metadata_json["status"] == "superseded"
    assert [m["content"] for m in env.service.search_memory(env.user, "PostgreSQL", db=env.db, conversation_id="one")] == ["PostgreSQL"]


def test_evidence_retry_cannot_boost_importance(env):
    for msg in range(1, 4):
        result = env.service.add_with_dedup(env.user, "Python", importance=.6,
            metadata={"entity": "stack", "conversation_id": "one", "source_message_id": msg, "source_quote": "Python"}, db=env.db)
    value = result["importance"]
    for _ in range(4):
        result = env.service.add_with_dedup(env.user, "Python", importance=.6,
            metadata={"entity": "stack", "conversation_id": "one", "source_message_id": 3, "source_quote": "Python"}, db=env.db)
    assert result["importance"] == value and result["metadata"]["evidence_count"] == 3


def test_read_and_write_denial_enforced_inside_services(env):
    from src.web_app.agent.runtime.chat_control import execution, ChatExecution
    old = add(env, "Python", category="answer_preference")
    run = AgentRun(user_id=env.user, conversation_id="one", memory_policy={"writes_blocked": True, "use_memory": False})
    env.db.add(run)
    env.db.commit()
    token = execution.set(ChatExecution(run.id))
    try:
        assert env.service.search_memory(env.user, "Python", db=env.db) == []
        assert env.service.get_baseline_memories(env.user, db=env.db) == []
        assert env.service.search_memory(env.user, "Python", db=env.db, management=True)[0]["id"] == old["id"]
        with pytest.raises(ValueError, match="memory_writes_disabled"):
            add(env, "Must not save")
    finally:
        execution.reset(token)


def test_source_ownership_and_model_cannot_expand_scope(env, monkeypatch):
    module, manager, run, job = setup_job(env, monkeypatch)
    saved = add(env, "本项目使用 Python", category="answer_preference", run_id=run.id, source_quote="本项目使用 Python", source_verified=True)
    assert saved["scope"] == "conversation"
    fake = add(env, "Fake personal preference", category="answer_preference", run_id=run.id, source_quote="助手建议使用 MySQL", source_verified=True)
    assert fake["status"] == "pending" and not fake["metadata"]["source_verified"]
    other = User(email="intruder@example.test", hashed_password="x")
    env.db.add(other)
    env.db.commit()
    with pytest.raises(ValueError, match="invalid_memory_source"):
        env.service.add_with_dedup(other.id, "stolen", metadata={"run_id": run.id}, db=env.db)


def test_closing_during_extraction_fences_final_write(env, monkeypatch):
    module, manager, run, job = setup_job(env, monkeypatch)
    from src.web_app.agent.llm.context import ModelExecutionContext
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_model_context", lambda *a: ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "test", "Test"))
    async def extract(*a, **kw):
        from src.web_app.services.profile_service import update_profile
        update_profile(env.db, env.user, {"generate_memory": False})
        return {"semantic_memories": [{"content": "Python", "importance": .95, "confidence": .95, "source_quote": "本项目使用 Python"}]}
    monkeypatch.setattr("src.web_app.memory.extractor.LlmMemoryExtractor.extract", extract)
    module.process(job.id)
    env.db.expire_all()
    assert env.db.get(MemoryMaintenanceTask, job.id).status == "cancelled"
    assert not list(env.db.scalars(select(Memory)))


def test_index_failure_is_durable_and_repaired(env, monkeypatch):
    from src.web_app.services import memory_tasks as module
    from src.web_app.services.memory_service import memory_service
    from src.web_app.core.config import settings
    monkeypatch.setattr(settings, "qdrant_url", "http://index.invalid")
    monkeypatch.setattr(settings, "enable_neo4j", False)
    factory = sessionmaker(bind=env.db.get_bind())
    monkeypatch.setattr(module, "SessionLocal", factory)
    class Store:
        available = False
        def upsert_memory(self, **kw):
            if not self.available:
                raise RuntimeError("outage")
            return "point-" + str(kw["memory_id"])
    store = Store()
    monkeypatch.setattr(MemoryService, "_get_qdrant_store", lambda self: store)
    saved = add(env, "Python")
    assert saved["ok"] and not saved["qdrant_indexed"]
    assert env.db.get(Memory, saved["id"]).metadata_json["index_pending"]
    store.available = True
    module.retry_indexes()
    env.db.expire_all()
    row = env.db.get(Memory, saved["id"])
    assert row.metadata_json["qdrant_indexed"] and not row.metadata_json["index_pending"]
    assert row.qdrant_point_id == "point-" + str(row.id)


@pytest.mark.parametrize("relation", ["duplicate", "supplement", "correction", "unrelated", "ambiguous"])
def test_structured_comparison_uses_validated_model_output(monkeypatch, relation):
    from src.web_app.memory.facts import compare
    monkeypatch.setattr("src.web_app.agent.llm.factory.get_chat_model", lambda *a, **k:
        SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content='{"relation":"' + relation + '","reason":"deterministic stub"}')))
    assert compare("old fact", "new fact", same_key=True) == relation


def test_structured_comparison_invalid_output_stays_pending(monkeypatch):
    from src.web_app.memory.facts import compare
    monkeypatch.setattr("src.web_app.agent.llm.factory.get_chat_model", lambda *a, **k:
        SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="not json")))
    assert compare("likes Python", "dislikes Python", same_key=True) == "ambiguous"


def test_supplement_and_different_entities(env, monkeypatch):
    original = add(env, "服务器 A 使用 Python", entity="server-a", source_message_id=1, source_quote="Python")
    monkeypatch.setattr("src.web_app.services.memory_service.compare", lambda *a, **k: "supplement")
    additional = add(env, "服务器 A 运行 Linux", entity="server-a", source_message_id=2, source_quote="Linux")
    assert "Python" in additional["content"] and "Linux" in additional["content"]
    assert additional["metadata"]["evidence_count"] == 2
    monkeypatch.setattr("src.web_app.services.memory_service.compare", lambda *a, **k: "unrelated")
    other = add(env, "服务器 B 使用 MySQL", entity="server-b")
    assert other["id"] != additional["id"] and other["status"] == "active"
    assert env.db.get(Memory, additional["id"]).metadata_json.get("status", "active") == "active"


def test_worker_restart_serializes_conversation_and_caps_concurrency(tmp_path, monkeypatch):
    import asyncio
    import threading
    import time
    from sqlalchemy import create_engine
    from src.web_app.db.base import Base
    from src.web_app.services import memory_tasks as module
    engine = create_engine("sqlite:///" + str(tmp_path / "worker.db"), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        user = User(email="worker@example.test", hashed_password="x")
        db.add(user)
        db.flush()
        uid = user.id
        for index, cid in enumerate(["one", "one", "two", "three"]):
            run = AgentRun(user_id=uid, conversation_id=cid, status="completed")
            db.add(run)
            db.flush()
            db.add(MemoryMaintenanceTask(user_id=uid, conversation_id=cid, run_id=run.id,
                source_message_id=index + 1, status="running" if index == 0 else "pending", attempts=1 if index == 0 else 0))
        db.commit()
    monkeypatch.setattr(module, "SessionLocal", factory)
    monkeypatch.setattr(module, "retry_indexes", lambda: None)
    lock = threading.Lock()
    active, seen, maximum = {}, [], [0]
    def fake_process(job_id):
        with factory() as db:
            job = db.get(MemoryMaintenanceTask, job_id)
            cid = job.conversation_id
            with lock:
                active[cid] = active.get(cid, 0) + 1
                assert active[cid] == 1
                maximum[0] = max(maximum[0], sum(active.values()))
                seen.append((cid, job.source_message_id))
            time.sleep(.03)
            with lock:
                job.status = "completed"
                db.commit()
                active[cid] -= 1
    monkeypatch.setattr(module, "process", fake_process)
    async def work():
        manager = module.MemoryTasks()
        manager.recover()
        manager.start(uid, "one")  # duplicate scheduling is coalesced
        await asyncio.wait_for(asyncio.gather(*tuple(manager.tasks.values())), timeout=8)
        await manager.shutdown()
    asyncio.run(work())
    assert maximum[0] == 2
    assert [msg for cid, msg in seen if cid == "one"] == [1, 2]
    with factory() as db:
        assert all(job.status == "completed" for job in db.scalars(select(MemoryMaintenanceTask)))
    engine.dispose()


def test_clearing_during_summary_does_not_restore_old_context(env, monkeypatch):
    from src.web_app.services.conversation_summary_service import ConversationSummaryService
    from src.web_app.services.agent_service import clear_conversation
    def summarize(prompt):
        clear_conversation(env.db, env.user, "one")
        return '{"summary_text":"stale", "decisions":["old choice"]}'
    monkeypatch.setattr("src.web_app.services.conversation_summary_service._llm_call", summarize)
    service = ConversationSummaryService()
    assert service.update_after_turn("one", env.user, [{"role":"user", "content":"old choice"}], db=env.db, last_message_id=2) is None
    assert service.get_summary("one", env.user, db=env.db) is None


def test_clearing_fences_old_run_manual_memory_tool(env, monkeypatch):
    from src.web_app.services.agent_service import clear_conversation
    from src.web_app.agent.runtime.chat_control import execution, ChatExecution
    module, manager, run, job = setup_job(env, monkeypatch)
    clear_conversation(env.db, env.user, "one")
    token = execution.set(ChatExecution(run.id))
    try:
        with pytest.raises(ValueError, match="memory_writes_disabled"):
            add(env, "Old turn cannot recreate this fact")
    finally:
        execution.reset(token)


def test_basics_auto_opt_in_preserves_default_confirmation(env):
    from src.web_app.memory.basic_facts import prepare
    state = {"user_id":env.user, "run_id":1, "conversation_id":"one", "user_input":"我叫常", "request":{}}
    assert not prepare(env.db, state)["authorized"]
    state["memory_policy"] = {"generate_memory":True}
    assert prepare(env.db, state)["authorized"]
    state["request"] = {"write_memory":False}
    assert not prepare(env.db, state)["authorized"]


def test_context_budgets_keep_each_packet_and_latest_constraints():
    from src.web_app.context.builder import ContextBuilder
    from src.web_app.context.packets import ContextConfig
    from src.web_app.memory.tokens import count
    payload = {"conversation_history":"Recent context " * 6000 + "LATEST_CORRECTION",
        "conversation_summary":"EARLY_DECISION " + "Summary " * 6000 + "UNFINISHED_TASK",
        "memory":"PERSONAL_PREFERENCE " + "Memory " * 6000 + "LATEST_PREFERENCE",
        "task":"CURRENT_REQUEST"}
    context, debug = ContextBuilder(ContextConfig(max_tokens=4000)).build_with_debug(payload)
    assert {"conversation_history", "conversation_summary", "memory", "task"} <= set(debug["selected_sources"])
    for text in ("CURRENT_REQUEST", "LATEST_CORRECTION", "EARLY_DECISION", "UNFINISHED_TASK", "PERSONAL_PREFERENCE", "LATEST_PREFERENCE"):
        assert text in context
    assert count(context) <= 4000


def test_token_chunks_roundtrip_multibyte_source_and_small_budgets():
    from src.web_app.memory.tokens import count, fit, chunks
    text = "中文来源证据😀：必须保留末尾约束。" * 300
    parts = chunks(text, budget=17)
    assert "".join(parts) == text and all(count(part) <= 17 for part in parts)
    for budget in (0, 1, 12, 50):
        assert count(fit(text, budget)) <= budget


def test_old_duplicate_evidence_does_not_rewind_latest_source(env):
    first = add(env, "Python", entity="stack", source_message_id=20, source_quote="Python")
    duplicate = add(env, "Python", entity="stack", source_message_id=10, source_quote="Python")
    assert duplicate["id"] == first["id"]
    assert duplicate["metadata"]["source_message_id"] == 20 and duplicate["metadata"]["evidence_count"] == 2
    correction = add(env, "MySQL", entity="stack", source_message_id=15, source_quote="MySQL")
    assert correction["status"] == "pending"


def test_candidate_comparison_continues_after_unrelated_hit(env, monkeypatch):
    monkeypatch.setattr("src.web_app.services.memory_service.compare", lambda *a, **k: "unrelated")
    a = add(env, "服务器 A 使用 Python", entity="a")
    b = add(env, "服务器 B 使用 MySQL", entity="b")
    def compare(old, new, **kw):
        return "correction" if "服务器 B" in old else "unrelated"
    monkeypatch.setattr("src.web_app.services.memory_service.compare", compare)
    new = add(env, "服务器 B 改用 PostgreSQL")
    assert new["metadata"]["supersedes"] == b["id"]
    assert env.db.get(Memory, a["id"]).metadata_json.get("status", "active") == "active"


def test_unknown_sources_and_low_confidence_are_not_activated(env):
    extraction = {"semantic_memories":[{"content":"model suggestion", "category":"answer_preference", "importance":.95, "confidence":.95}]}
    result = env.service._save_extracted(env.user, extraction, env.db, "")
    assert result["saved"]["semantic"][0]["status"] == "pending"
    assert not env.service.search_memory(env.user, "model", db=env.db)


def test_conversation_settings_api_can_reset_and_rejects_other_users(env):
    from fastapi import HTTPException
    from src.web_app.api.v1.agent import MemorySettingsRequest, get_memory_settings, set_memory_settings
    response = set_memory_settings("one", MemorySettingsRequest(generate_memory=True, use_memory=False), env.user, env.db)["data"]
    assert response["generate_memory"] and not response["use_memory"]
    response = set_memory_settings("one", MemorySettingsRequest(generate_memory=None, use_memory=None), env.user, env.db)["data"]
    assert response["overrides"] == {} and response["use_memory"] and not response["generate_memory"]
    with pytest.raises(HTTPException) as error:
        get_memory_settings("one", env.user + 1, env.db)
    assert error.value.status_code == 404


def test_management_keeps_hidden_history_and_scope_filter(env):
    from src.web_app.db.repositories.memory_repository import MemoryRepository
    hidden = add(env, "hidden fact")
    item = env.db.get(Memory, hidden["id"])
    item.metadata_json = {**item.metadata_json, "visible_in_long_term_memory":False}
    env.db.commit()
    repo = MemoryRepository(env.db)
    assert repo.list_long_term(env.user)[1] == 0
    assert repo.list_long_term(env.user, status="all", scope="conversation", scope_id="one")[1] == 1
    assert repo.list_long_term(env.user, status="all", scope="user")[1] == 0
    assert not env.service.search_memory(env.user, "hidden", db=env.db, conversation_id="one")


def test_personal_scope_requires_personal_statement_for_model_sources():
    assert infer_scope({"run_id":1, "category":"answer_preference", "source_quote":"请回答这个问题"}, "one") == ("conversation", "one")
    assert infer_scope({"run_id":1, "category":"preference", "source_quote":"以后接入 PostgreSQL", "source_verified":True, "personal_long_term":True}, "one") == ("conversation", "one")
    assert infer_scope({"run_id":1, "category":"answer_preference", "source_quote":"以后请用中文回答"}, "one") == ("user", "")
    assert infer_scope({"run_id":1, "category":"preference", "source_quote":"我一直喜欢喝茶", "source_verified":True, "personal_long_term":True}, "one") == ("user", "")


def test_recovery_closes_reply_enqueue_crash_window_without_rebuilding_history(env, monkeypatch):
    module, manager, run, job = setup_job(env, monkeypatch)
    env.db.delete(job)
    legacy = AgentRun(user_id=env.user, conversation_id="one", status="completed", memory_policy={})
    env.db.add(legacy)
    env.db.commit()
    manager._recover_missing()
    env.db.expire_all()
    jobs = list(env.db.scalars(select(MemoryMaintenanceTask)))
    assert len(jobs) == 1 and jobs[0].run_id == run.id
    assert not any(item.run_id == legacy.id for item in jobs)


def test_legacy_memory_api_defaults_remain_manageable_without_chat_recall(env):
    from src.web_app.api.v1.memory import add_memory, search_memory
    saved = add_memory({"content":"Legacy working note"}, env.user, env.db)["data"]
    assert saved["ok"] and saved["memory_type"] == "working"
    results = search_memory({"query":"Legacy"}, env.user, env.db)["data"]
    assert [item["id"] for item in results] == [saved["id"]]
    assert not env.service.search_memory(env.user, "Legacy", db=env.db)
