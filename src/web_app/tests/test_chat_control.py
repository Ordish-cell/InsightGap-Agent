import asyncio
from types import SimpleNamespace
from langchain_core.messages import AIMessageChunk

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.web_app.agent.runtime.chat_control import ChatExecution, capabilities, controlled_node, execution
from src.web_app.agent.runtime.event_ledger import publish_event
from src.web_app.db.base import Base
from src.web_app.models.orm import AgentChatMessage, AgentConversation, AgentEvent, AgentRun, AgentRunControl, User
from src.web_app.services.agent_run_task_manager import AgentRunTaskManager
from src.web_app.services.chat_control_service import ChatControlError, continuation_context, recover_chat_controls, request_control


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'chat.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    manager = AgentRunTaskManager()
    monkeypatch.setattr("src.web_app.services.agent_run_task_manager.SessionLocal", factory)
    monkeypatch.setattr("src.web_app.services.agent_run_task_manager.agent_run_task_manager", manager)
    with factory() as db:
        user = User(email="chat-control@example.test", hashed_password="x")
        db.add(user)
        db.flush()
        run = AgentRun(user_id=user.id, conversation_id="chat", thread_id="chat-thread", user_input="解释架构",
                       status="created", chat_control_phase="enabled", graph_state={"model_context": {}})
        db.add(run)
        db.flush()
        conversation = AgentConversation(user_id=user.id, conversation_id="chat", last_run_id=run.id)
        db.add(conversation)
        for role, content in (("user", "解释架构"), ("assistant", "")):
            db.add(AgentChatMessage(message_id=f"original-{role}", user_id=user.id, conversation_id="chat",
                                   thread_id="chat-thread", run_id=run.id, role=role, content=content, status="thinking" if role == "assistant" else "completed"))
        db.commit()
        ids = user.id, run.id
    yield SimpleNamespace(factory=factory, manager=manager, user=ids[0], run=ids[1])
    engine.dispose()


async def wait_controls(env):
    await asyncio.wait_for(asyncio.gather(*list(env.manager._controls.values())), 4)


def fake_execution(env, monkeypatch, *, delta="已经输出", hold_successor=False):
    ready = asyncio.Event()
    captured = []

    async def fake(db, user_id, run_id, payload):
        run = db.get(AgentRun, run_id)
        run.status = "running"
        db.commit()
        if run_id == env.run:
            if delta:
                publish_event(db, None, run_id, "answer_delta", {"text": delta}, user_id=user_id)
            ready.set()
            await asyncio.Event().wait()
        else:
            captured.append(continuation_context(db, run))
            if hold_successor:
                await asyncio.Event().wait()
            run.status = "completed"
            db.commit()
    monkeypatch.setattr("src.web_app.services.agent_service.execute_prepared_run", fake)
    env.manager.start(env.run, env.user, payload={"_chat_managed": True})
    return ready, captured


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", ["", "已经输出"])
async def test_stop_before_first_token_and_midstream(env, monkeypatch, delta):
    ready, _ = fake_execution(env, monkeypatch, delta=delta)
    await ready.wait()
    with env.factory() as db:
        result = request_control(db, env.user, env.run, "interrupt", "stop-1")
        assert result["status"] == "accepted"
        assert result["successor_run_id"] is None
    await wait_controls(env)
    with env.factory() as db:
        run = db.get(AgentRun, env.run)
        assert run.status == "interrupted"
        assert run.final_answer == delta
        message = db.execute(select(AgentChatMessage).where(AgentChatMessage.role == "assistant")).scalar_one()
        assert message.content == delta
        assert message.status == "interrupted"
        assert not capabilities(run)["can_interrupt"]
        assert db.execute(select(AgentRunControl.status)).scalar_one() == "applied"
    assert not env.manager.is_running(env.run)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["只讲聊天模块，简单一点", "换个话题，解释光合作用"])
async def test_steer_persists_once_and_starts_after_cleanup(env, monkeypatch, text):
    ready, captured = fake_execution(env, monkeypatch)
    await ready.wait()
    with env.factory() as db:
        first = request_control(db, env.user, env.run, "steer", "same-request", text)
        second = request_control(db, env.user, env.run, "steer", "same-request", text)
        assert first == second
        successor = db.get(AgentRun, first["successor_run_id"])
        assert successor.status == "queued"
        assert len(db.execute(select(AgentChatMessage)).scalars().all()) == 4
    await wait_controls(env)
    await asyncio.sleep(0)
    assert len(captured) == 1
    assert "解释架构" in captured[0] and text in captured[0] and "已经输出" in captured[0]
    assert "unfinished" in captured[0]
    with env.factory() as db:
        assert db.get(AgentRun, env.run).status == "interrupted"
        assert request_control(db, env.user, env.run, "steer", "same-request", text)["status"] == "applied"
        with pytest.raises(ChatControlError, match="已接续"):
            request_control(db, env.user, env.run, "steer", "different-request", "another")
    await env.manager.shutdown()


@pytest.mark.asyncio
async def test_continuous_steering_preserves_order(env, monkeypatch):
    ready, captured = fake_execution(env, monkeypatch, hold_successor=True)
    await ready.wait()
    with env.factory() as db:
        first = request_control(db, env.user, env.run, "steer", "one", "只讲聊天")
    await wait_controls(env)
    await asyncio.sleep(0)
    with env.factory() as db:
        second = request_control(db, env.user, first["successor_run_id"], "steer", "two", "再简短一点")
    await wait_controls(env)
    await asyncio.sleep(0)
    assert captured[-1].index("解释架构") < captured[-1].index("只讲聊天") < captured[-1].index("再简短一点")
    assert second["successor_run_id"] != first["successor_run_id"]
    await env.manager.shutdown()


@pytest.mark.asyncio
async def test_boundary_wins_or_cancel_wins_never_both(env, monkeypatch):
    ready, _ = fake_execution(env, monkeypatch)
    await ready.wait()
    token = env.manager._tokens[env.run]
    called = []
    async def tool(state):
        called.append("tool")
        return state
    with env.factory() as db:
        context = execution.set(token)
        try:
            await controlled_node("tool_runtime", tool, db)({"run_id": env.run})
        finally:
            execution.reset(context)
        with pytest.raises(ChatControlError):
            request_control(db, env.user, env.run, "interrupt", "blocked")
    assert called == ["tool"]
    await env.manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_fences_late_events_and_side_effect_entry(env, monkeypatch):
    ready, _ = fake_execution(env, monkeypatch)
    await ready.wait()
    token = env.manager._tokens[env.run]
    with env.factory() as db:
        request_control(db, env.user, env.run, "interrupt", "stop")
        context = execution.set(token)
        try:
            with pytest.raises(asyncio.CancelledError):
                publish_event(db, None, env.run, "answer_delta", {"text": "late"}, user_id=env.user)
            async def forbidden(state):
                pytest.fail("Side effect entered after control was accepted")
            with pytest.raises(asyncio.CancelledError):
                await controlled_node("research_agent", forbidden, db)({"run_id": env.run})
        finally:
            execution.reset(context)
    await wait_controls(env)


@pytest.mark.asyncio
async def test_completed_run_steer_and_interrupt_are_safe(env, monkeypatch):
    async def fake(*args):
        return None
    monkeypatch.setattr("src.web_app.services.agent_service.execute_prepared_run", fake)
    with env.factory() as db:
        db.get(AgentRun, env.run).status = "completed"
        db.commit()
        result = request_control(db, env.user, env.run, "steer", "next", "继续")
        assert result["successor_run_id"]
    await wait_controls(env)
    await env.manager.shutdown()


@pytest.mark.asyncio
async def test_controls_api_validates_owner_input_and_disabled_phase(env):
    from src.web_app.api.v1.agent import router
    from src.web_app.db.session import get_db
    from src.web_app.services.auth_service import get_current_user_id
    app = FastAPI()
    app.include_router(router)
    def db_override():
        with env.factory() as db:
            yield db
    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_current_user_id] = lambda: env.user + 1
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post(f"/runs/{env.run}/interrupt", json={"client_command_id": "x"})).status_code == 404
        app.dependency_overrides[get_current_user_id] = lambda: env.user
        assert (await client.post(f"/runs/{env.run}/steer", json={"client_command_id": "x", "text": " "})).status_code == 422
        assert (await client.post(f"/runs/{env.run}/steer", json={"client_command_id": "x", "text": "hi", "attachment_ids": [1]})).status_code == 422
        with env.factory() as db:
            db.get(AgentRun, env.run).chat_control_phase = "disabled"
            db.commit()
        assert (await client.post(f"/runs/{env.run}/interrupt", json={"client_command_id": "x"})).status_code == 409


def test_restart_preserves_partial_and_does_not_touch_approval(env):
    with env.factory() as db:
        publish_event(db, None, env.run, "answer_delta", {"text": "部分内容"}, user_id=env.user)
        db.add(AgentRun(user_id=env.user, conversation_id="other", status="waiting_approval", chat_control_phase="disabled"))
        db.commit()
    recover_chat_controls(env.factory)
    recover_chat_controls(env.factory)
    with env.factory() as db:
        assert db.get(AgentRun, env.run).final_answer == "部分内容"
        assert db.execute(select(AgentRun).where(AgentRun.conversation_id == "other")).scalar_one().status == "waiting_approval"
        assert len(db.execute(select(AgentEvent).where(AgentEvent.event_type == "run_interrupted")).scalars().all()) == 1


@pytest.mark.asyncio
async def test_intent_call_is_cancellable_before_any_answer(env, monkeypatch):
    from src.web_app.agent.runtime import nodes as supervisor_nodes
    entered, closed = asyncio.Event(), asyncio.Event()
    class Model:
        def bind_tools(self, tools):
            return self
        async def astream(self, prompt):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield AIMessageChunk(content="")
            finally:
                closed.set()
    monkeypatch.setattr(supervisor_nodes, "get_chat_model", lambda *a, **k: Model())
    with env.factory() as db:
        task = asyncio.create_task(supervisor_nodes.SupervisorNodes(db, {}).model_turn(
            {"run_id": env.run, "thread_id": "chat", "user_id": env.user, "user_input": "hello", "runtime_budget": {"steps": 1}}))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()


def test_migration_generates_only_incremental_sql():
    import importlib.util
    import io
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    path = Path(__file__).resolve().parents[3] / "alembic/versions/20260907_0014_chat_control.py"
    spec = importlib.util.spec_from_file_location("chat_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = io.StringIO()
    context = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output})
    with Operations.context(context):
        module.upgrade()
    sql = output.getvalue()
    assert "CREATE TABLE agent_run_controls" in sql
    assert "ALTER TABLE agent_runs ADD COLUMN" in sql
    assert "DELETE FROM" not in sql and "DROP TABLE" not in sql


@pytest.mark.asyncio
async def test_real_stream_adapter_closes_on_cancel(env, monkeypatch):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.runtime import nodes as mod
    entered, closed = asyncio.Event(), asyncio.Event()
    class Model:
        def bind_tools(self, tools):
            return self
        async def astream(self, prompt):
            try:
                yield AIMessageChunk(content=[{"type": "reasoning", "summary": [{"text": "private reasoning"}]}])
                yield AIMessageChunk(content=[{"type": "text", "text": "真实流适配器的部分输出"}])
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
    monkeypatch.setattr(mod, "get_chat_model", lambda *a, **k: Model())
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        task = asyncio.create_task(nodes.model_turn({**{"run_id": env.run, "user_id": env.user, "user_input": "解释架构"}, "runtime_budget": {"steps": 1}}))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        deltas = db.execute(select(AgentEvent).where(AgentEvent.run_id == env.run, AgentEvent.event_type == "agent_text_delta")).scalars().all()
        assert [event.payload_json["text"] for event in deltas] == ["真实流适配器的部分输出"]


@pytest.mark.asyncio
async def test_cancellation_timeout_never_starts_successor(env, monkeypatch):
    ready, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr("src.web_app.services.agent_run_task_manager.CHAT_CANCEL_TIMEOUT_SECONDS", 0.02)
    async def stubborn(db, user_id, run_id, payload):
        assert run_id == env.run, "Timed-out cancellation started a successor"
        ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise
    monkeypatch.setattr("src.web_app.services.agent_service.execute_prepared_run", stubborn)
    env.manager.start(env.run, env.user, payload={"_chat_managed": True})
    await ready.wait()
    with env.factory() as db:
        command = request_control(db, env.user, env.run, "steer", "timeout", "保留这条新消息")
    await wait_controls(env)
    with env.factory() as db:
        assert db.execute(select(AgentRunControl.status)).scalar_one() == "failed"
        successor = db.get(AgentRun, command["successor_run_id"])
        assert successor.status == "failed"
        assert successor.user_input == "保留这条新消息"
    release.set()
    await env.manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_immediately_after_start(env, monkeypatch):
    async def forbidden(*args):
        pytest.fail("Cancelled-before-start execution entered")
    monkeypatch.setattr("src.web_app.services.agent_service.execute_prepared_run", forbidden)
    env.manager.start(env.run, env.user, payload={"_chat_managed": True})
    with env.factory() as db:
        request_control(db, env.user, env.run, "interrupt", "immediate")
    await wait_controls(env)
    with env.factory() as db:
        assert db.get(AgentRun, env.run).status == "interrupted"


@pytest.mark.asyncio
async def test_full_chat_service_continuation_and_final_snapshot(env, monkeypatch):
    from src.web_app.agent.llm.context import ModelExecutionContext
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.runtime import nodes as mod
    from src.web_app.services.agent_service import get_run
    context = ModelExecutionContext(1, 1, 1, "test", "openai_chat_completions", "test", "test")
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_run_model_context", lambda *a: context)
    monkeypatch.setattr("src.web_app.services.agent_service._update_conversation_summary_after_turn", lambda **k: None)
    entered = asyncio.Event()
    prompts = []
    class Model:
        def bind_tools(self, tools):
            return self
        async def astream(self, prompt):
            prompt = "\n".join(message.content for message in prompt)
            prompts.append(prompt)
            if "unfinished" not in prompt:
                yield AIMessageChunk(content="架构包括聊天与其他模块。")
                entered.set()
                await asyncio.Event().wait()
            else:
                assert "只讲聊天模块" in prompt and "unfinished" in prompt
                yield AIMessageChunk(content="聊天模块：接收消息、构建上下文、流式回复。")
    monkeypatch.setattr(mod, "get_chat_model", lambda *a, **k: Model())
    async def runtime_run(runtime, state):
        nodes = SupervisorNodes(runtime.db, runtime.payload)
        await nodes.permission_guard(state)
        await nodes.bootstrap_context(state)
        result = await nodes.supervisor(state)
        answer = result["final_answer"]
        return {**state, "status": "completed", "final_answer": answer, "final_output": answer,
                "_answer_delta_emitted": True, "route_plan": {"intent": "chat", "route": ["final_response"]}}
    monkeypatch.setattr("src.web_app.services.agent_service.AgentRuntime.run", runtime_run)
    env.manager.start(env.run, env.user, payload={"_chat_managed": True})
    await asyncio.wait_for(entered.wait(), 3)
    with env.factory() as db:
        result = request_control(db, env.user, env.run, "steer", "full-service", "只讲聊天模块")
    await wait_controls(env)
    successor_task = env.manager._tasks.get(result["successor_run_id"])
    if successor_task:
        await asyncio.wait_for(successor_task, 3)
    with env.factory() as db:
        successor = db.get(AgentRun, result["successor_run_id"])
        assert successor.status == "completed"
        assert successor.graph_state["model_context"]["model_config_id"] == 1
        assert successor.final_answer == "聊天模块：接收消息、构建上下文、流式回复。"
        snapshot = get_run(db, env.user, successor.id)
        assert snapshot["controls"][-1]["status"] == "applied"
        assert not snapshot["can_steer"]
        assert db.get(AgentRun, env.run).status == "interrupted"
    assert len(prompts) == 2
