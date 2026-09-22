"""Basic facts share real tool receipts, PostgreSQL-compatible storage and chat projection."""
import json
import pytest
from sqlalchemy import select
from langchain_core.messages import AIMessageChunk

from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_native_supervisor import Model
from src.web_app.tests.test_supervisor_loop import state
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.agent.runtime.graph_builder import build_graph
from src.web_app.agent.runtime.hooks import save_outputs
from src.web_app.agent.runtime.finalization import finish, recover_answer
from src.web_app.memory.basic_facts import parse_facts, prepare
from src.web_app.models.orm import AgentRun, AgentChatMessage, AgentConversation, Memory, ToolCall, User
from src.web_app.services.memory_service import memory_service
from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.api.v1.memory import growth_profile


@pytest.fixture(autouse=True)
def no_indexes(monkeypatch):
    monkeypatch.setattr(memory_service, "_get_qdrant_store", lambda: None)
    monkeypatch.setattr(memory_service, "_sync_memory_graph", lambda *a: {})


@pytest.mark.parametrize("text", [
    "这次用中文回答", "他说我叫常", '“我叫常”', "假如我叫常", "我叫什么？",
    "我叫常吗", "我叫常的话", "不要记住，我叫常", "以后用中文回答可以吗？",
    "> 我叫常", "```我叫常```", "例如我叫常", "附件说我叫常", "我叫常，不要保存",
    "我叫常，帮我查资料", "我叫常。假设以后用英语回答", "我叫常，我叫李", "我叫你过来", "我叫他小常", "我叫常是个例子",
])
def test_parser_does_not_guess(text):
    assert parse_facts(text)[0] == []


@pytest.mark.parametrize("text,key,value,explicit", [
    ("我叫常", "preferred_name", "常", False),
    ("以后叫我小常", "preferred_name", "小常", False),
    ("记住，我叫常", "preferred_name", "常", True),
    ("以后用中文回答", "response_language", "中文", False),
    ("以后用英文回答", "response_language", "英语", False),
    ("以后用繁体字", "script_preference", "繁体", False),
])
def test_parser_clear_statements(text, key, value, explicit):
    assert parse_facts(text) == ([{"key": key, "value": value}], explicit)


async def turn(env, monkeypatch, text, conversation="chat", **request):
    model = Model([[AIMessageChunk(content="好的。")]])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    with env.factory() as db:
        if not db.scalar(select(AgentConversation).where(AgentConversation.conversation_id == conversation)):
            db.add(AgentConversation(user_id=env.user, conversation_id=conversation))
        run = AgentRun(user_id=env.user, conversation_id=conversation, user_input=text)
        db.add(run)
        db.commit()
        s = {**state(env), "run_id": run.id, "conversation_id": conversation, "user_input": text, "request": request}
        nodes = SupervisorNodes(db, request)
        result = await build_graph(nodes).ainvoke(s)
        public = json.loads(json.dumps(result["final_payload"], default=str))
        run.status = result["status"]
        run.final_response = public
        db.add(AgentChatMessage(message_id=f"assistant-{run.id}", user_id=env.user, run_id=run.id,
            conversation_id=conversation, role="assistant", content=result["final_answer"],
            status=result["status"], metadata_json={"final_response": public}))
        db.commit()
        assert len(model.requests) == 1  # no extraction/model round trip
        result["test_prompt"] = str(model.requests[0])
        return result


@pytest.mark.asyncio
async def test_confirm_cross_session_rename_archive_delete(env, monkeypatch):
    first = await turn(env, monkeypatch, "我叫常")
    assert "是否" in first["final_answer"] and first["final_payload"]["memory_proposal"]
    with env.factory() as db:
        assert not MemoryRepository(db).list_by_user(env.user)
    confirmed = await turn(env, monkeypatch, "确认记住")
    assert "已保存跨会话记忆" in confirmed["final_answer"]
    assert confirmed["memory_updates"] and not confirmed["save_policy"].get("write_memory")
    with env.factory() as db:
        memories = memory_service.get_baseline_memories(env.user, db=db)
        assert [m["metadata"]["fact_value"] for m in memories] == ["常"]
        assert MemoryRepository(db).list_long_term(env.user)[1] == 1
        assert db.scalar(select(ToolCall).where(ToolCall.run_id == confirmed["run_id"]))
        replay = recover_answer(SupervisorNodes(db, {}), {**state(env), "run_id": first["run_id"]})
        assert replay["final_payload"]["memory_proposal"] == first["final_payload"]["memory_proposal"]
        assert replay["final_answer"] == first["final_answer"]
    second_session = await turn(env, monkeypatch, "我的称呼是什么", conversation="new-session")
    assert "用户确认的称呼：常" in second_session["test_prompt"]
    await turn(env, monkeypatch, "确认记住", conversation="new-session")
    await turn(env, monkeypatch, "以后叫我小常")
    changed = await turn(env, monkeypatch, "确认记住")
    with env.factory() as db:
        rows = MemoryRepository(db).list_by_user(env.user)
        assert len(rows) == 2
        active = memory_service.get_baseline_memories(env.user, db=db)
        assert [m["metadata"]["fact_value"] for m in active] == ["小常"]
        assert any(m.metadata_json["status"] == "superseded" for m in rows)
        row = db.get(Memory, active[0]["id"])
        row.metadata_json = {**row.metadata_json, "status": "archived"}
        db.commit()
        assert memory_service.get_baseline_memories(env.user, db=db) == []
        memory_service.forget_memory(env.user, row.id, db)
        assert memory_service.get_baseline_memories(env.user, db=db) == []
    assert "已保存" in changed["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["不要", "聊点别的", "这次用中文回答"])
async def test_topic_change_expires_proposal(env, monkeypatch, reply):
    await turn(env, monkeypatch, "我叫常")
    await turn(env, monkeypatch, reply)
    await turn(env, monkeypatch, "确认记住")
    with env.factory() as db:
        assert MemoryRepository(db).list_by_user(env.user) == []


@pytest.mark.asyncio
async def test_direct_explicit_false_and_repeated_value(env, monkeypatch):
    await turn(env, monkeypatch, "记住，我叫常", write_memory=False)
    with env.factory() as db:
        assert MemoryRepository(db).list_by_user(env.user) == []
    result = await turn(env, monkeypatch, "记住，我叫常")
    await turn(env, monkeypatch, "记住，我叫常")
    with env.factory() as db:
        assert len(MemoryRepository(db).list_by_user(env.user)) == 1
        # Same action replay reuses durable receipt.
        result.pop("_answer_completed_emitted", None)
        await save_outputs(SupervisorNodes(db, {}), result, "answer")
        assert len(MemoryRepository(db).list_by_user(env.user)) == 1


@pytest.mark.asyncio
async def test_short_confirmation_and_research_ambiguity(env, monkeypatch):
    await turn(env, monkeypatch, "我叫常")
    with env.factory() as db:
        msg = db.scalars(select(AgentChatMessage).order_by(AgentChatMessage.id.desc())).first()
        final = {**msg.metadata_json["final_response"], "research_proposal": {"query": "research"}}
        msg.metadata_json = {"final_response": final}
        db.commit()
        s = {**state(env), "run_id": 999, "user_input": "好"}
        assert not prepare(db, s)["authorized"]
        s["user_input"] = "确认记住"
        assert prepare(db, s)["authorized"]
    await turn(env, monkeypatch, "确认记住")
    await turn(env, monkeypatch, "以后用繁体字")
    short = await turn(env, monkeypatch, "好")
    assert "已保存" in short["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["error", "termination_reason", "writes_denied", "approval_required"])
async def test_incomplete_run_never_saves(env, flag):
    with env.factory() as db:
        s = {**state(env), "user_input": "记住，我叫常", flag: True}
        prepare(db, s)
        await save_outputs(SupervisorNodes(db, {}), s, "answer")
        assert MemoryRepository(db).list_by_user(env.user) == []
        assert not s.get("basic_memory_note")


def test_visibility_counts_priority_and_stale_vectors(env, monkeypatch):
    with env.factory() as db:
        for i in range(22):
            db.add(Memory(user_id=env.user, content=f"legacy-{i}", memory_type="semantic", importance=1,
                          metadata_json={"category": "tech_stack"}))
        other = User(email="other@example.test", hashed_password="x")
        db.add(other)
        db.commit()
        fact = {"key": "preferred_name", "value": "常"}
        saved = memory_service.save_basic_fact(env.user, fact, {}, db)
        assert not saved["qdrant_indexed"] and saved["ok"]
        db.add(Memory(user_id=env.user, content="hidden", memory_type="semantic", importance=1,
                      metadata_json={"visible_in_long_term_memory": False, "category": "tech_stack"}))
        db.commit()
        assert MemoryRepository(db).list_long_term(env.user)[1] == 23
        assert growth_profile(env.user, db)["data"]["semantic_count"] == 23
        assert memory_service.get_baseline_memories(env.user, db=db)[0]["id"] == saved["id"]
        assert len(memory_service.get_baseline_memories(env.user, db=db)) == 6
        assert not memory_service.get_baseline_memories(other.id, db=db)
        class StaleStore:
            def search_memory(self, **kwargs):
                return [{"memory_id": saved["id"], "score": 1}]
        monkeypatch.setattr(memory_service, "_get_qdrant_store", lambda: StaleStore())
        row = db.get(Memory, saved["id"])
        for status in ("archived", "superseded", "deleting", "low_confidence"):
            row.metadata_json = {**row.metadata_json, "status": status}
            db.commit()
            assert memory_service.search_memory(env.user, "常", db=db) == []
            assert saved["id"] not in [m["id"] for m in memory_service.get_baseline_memories(env.user, db=db)]
        assert memory_service.search_memory(other.id, "常", db=db) == []
        row.metadata_json = {**row.metadata_json, "status": "archived"}
        db.commit()
        assert MemoryRepository(db).list_long_term(env.user, status="all")[1] == 23
        assert MemoryRepository(db).list_long_term(env.user)[1] == 22


def test_transaction_failure_keeps_previous_value(env, monkeypatch):
    with env.factory() as db:
        first = memory_service.save_basic_fact(env.user, {"key": "preferred_name", "value": "常"}, {}, db)
        def fail():
            raise RuntimeError("database unavailable")
        with monkeypatch.context() as patch:
            patch.setattr(db, "commit", fail)
            with pytest.raises(RuntimeError):
                memory_service.save_basic_fact(env.user, {"key": "preferred_name", "value": "小常"}, {}, db)
        assert [m["id"] for m in memory_service.get_baseline_memories(env.user, db=db)] == [first["id"]]


@pytest.mark.asyncio
async def test_tool_save_failure_never_claims_success(env, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(memory_service, "save_basic_fact", fail)
    result = await turn(env, monkeypatch, "记住，我叫常")
    assert "已保存跨会话记忆" not in result["final_answer"]
    assert "未确认保存成功" in result["final_answer"]
    assert not result.get("memory_updates")


def test_restore_keeps_one_active_value_and_generic_dedup_skips_facts(env):
    with env.factory() as db:
        old = memory_service.save_basic_fact(env.user, {"key": "preferred_name", "value": "常"}, {}, db)
        new = memory_service.save_basic_fact(env.user, {"key": "preferred_name", "value": "小常"}, {}, db)
        memory_service.restore_memory(env.user, db.get(Memory, old["id"]), db)
        assert [m["id"] for m in memory_service.get_baseline_memories(env.user, db=db)] == [old["id"]]
        assert db.get(Memory, new["id"]).metadata_json["status"] == "superseded"
        generic = memory_service.add_with_dedup(env.user, old["content"], "semantic", .9, db=db)
        assert generic["id"] not in {old["id"], new["id"]}


def test_qdrant_failure_does_not_rollback_fact(env, monkeypatch):
    class BrokenStore:
        def upsert_memory(self, **kwargs):
            raise OSError("index unavailable")
    monkeypatch.setattr(memory_service, "_get_qdrant_store", lambda: BrokenStore())
    with env.factory() as db:
        saved = memory_service.save_basic_fact(env.user, {"key": "response_language", "value": "中文"}, {}, db)
        assert saved["ok"] and not saved["qdrant_indexed"] and saved["error"]
        assert memory_service.get_baseline_memories(env.user, db=db)[0]["id"] == saved["id"]


@pytest.mark.asyncio
async def test_cancelled_execution_never_saves(env):
    import asyncio
    from src.web_app.agent.runtime.chat_control import ChatExecution, execution
    with env.factory() as db:
        s = {**state(env), "user_input": "记住，我叫常", "runtime_budget": {"tool_calls": 0}}
        prepare(db, s)
        token = execution.set(ChatExecution(run_id=env.run, cancelled=True))
        try:
            with pytest.raises(asyncio.CancelledError):
                await save_outputs(SupervisorNodes(db, {}), s, "answer")
        finally:
            execution.reset(token)
        assert MemoryRepository(db).list_by_user(env.user) == []


@pytest.mark.parametrize("memory_type,metadata,visible", [
    ("semantic", {}, True), ("episodic", {}, True), ("working", {}, False),
    ("semantic", {"visible_in_long_term_memory": False}, False),
    ("working", {"visible_in_long_term_memory": True}, False),
])
def test_new_memory_defaults(env, memory_type, metadata, visible):
    with env.factory() as db:
        saved = memory_service.add_memory(env.user, "test", memory_type, .8, metadata=metadata, db=db)
        assert saved["metadata"]["status"] == "active"
        assert saved["metadata"]["visible_in_long_term_memory"] is visible


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["unfinished", "foreign_source", "wrong_conversation"])
async def test_only_owned_completed_previous_proposal_can_be_confirmed(env, monkeypatch, variant):
    await turn(env, monkeypatch, "我叫常")
    with env.factory() as db:
        msg = db.scalars(select(AgentChatMessage).order_by(AgentChatMessage.id.desc())).first()
        source = db.get(AgentRun, msg.run_id)
        if variant == "unfinished":
            msg.status = "interrupted"
        elif variant == "wrong_conversation":
            source.conversation_id = "other-chat"
        else:
            other = User(email="foreign@example.test", hashed_password="x")
            db.add(other)
            db.flush()
            source.user_id = other.id
        db.commit()
        proposal = prepare(db, {**state(env), "run_id": 999, "user_input": "确认记住"})
        assert not proposal["authorized"] and not proposal["facts"]


def test_low_confidence_and_unconfirmed_fixed_categories_not_preloaded(env):
    with env.factory() as db:
        for metadata in ({"category": "preferred_name"},
                         {"category": "tech_stack", "confidence": .2},
                         {"category": "tech_stack", "status": "deleting"}):
            db.add(Memory(user_id=env.user, content="not eligible", memory_type="semantic", importance=1,
                          metadata_json=metadata))
        db.commit()
        assert memory_service.get_baseline_memories(env.user, db=db) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted", "waiting_approval", "terminated"])
async def test_terminal_status_never_appends_save(env, status):
    with env.factory() as db:
        s = {**state(env), "status": status, "user_input": "记住，我叫常"}
        prepare(db, s)
        await save_outputs(SupervisorNodes(db, {}), s, "answer")
        assert MemoryRepository(db).list_by_user(env.user) == []
