import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src.web_app.agent.runtime.chat_control import ChatExecution, controlled_node, execution
from src.web_app.agent.runtime.chat_fast_path import RouteHeader, RouteProtocolError, chat_entry
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.models.orm import AgentEvent
from src.web_app.tests.test_chat_control import env


@pytest.mark.parametrize("chunks,route,body", [(["ch", "at\n你", "好"], "chat", "你好"),
    (["workflow\n"], "workflow", ""), (["chat\r\nanswer"], "chat", "answer")])
def test_header_split(chunks, route, body):
    parser = RouteHeader()
    assert "".join(parser.feed(chunk) for chunk in chunks) == body
    assert parser.route == route


@pytest.mark.parametrize("value", ["x" * 513, "other\n", "```chat\n"])
def test_invalid_header(value):
    with pytest.raises(RouteProtocolError):
        RouteHeader().feed(value)


def fake_model(monkeypatch, chunks, *, wait=False, error=False):
    calls, closed = [], []
    ready = asyncio.Event()
    async def stream(prompt):
        calls.append(prompt)
        try:
            for chunk in chunks:
                yield SimpleNamespace(content=chunk)
            ready.set()
            if error:
                raise RuntimeError("provider unavailable")
            if wait:
                await asyncio.Event().wait()
        finally:
            closed.append(True)
    monkeypatch.setattr("src.web_app.agent.runtime.chat_fast_path.get_chat_model", lambda *a, **k: SimpleNamespace(astream=stream))
    monkeypatch.setattr("src.web_app.agent.runtime.chat_fast_path.settings.chat_fast_path_enabled", True)
    return calls, closed, ready


def initial(env):
    return {"run_id": env.run, "user_id": env.user, "conversation_id": "chat", "thread_id": "chat-thread",
            "user_input": "唉，我好累", "interaction_version": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["唉，我好累", "你好", "解释一下哈希表"])
async def test_compiled_graph_chat_is_one_call(env, monkeypatch, question):
    from src.web_app.agent.runtime.graph import AgentRuntime
    calls, closed, _ = fake_model(monkeypatch, [[{"type": "reasoning", "text": "private"}], "ch", "at\n先休息", "一下。"])
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    with env.factory() as db:
        runtime = AgentRuntime(db, {})
        async def forbidden(state):
            pytest.fail("Chat entered heavy workflow")
        for name in ("home_intent_react", "planner", "parallel_prefetch", "parallel_read_stage", "final_response"):
            monkeypatch.setattr(runtime.nodes, name, forbidden)
        state = {**initial(env), "user_input": question}
        result = await runtime.run(state)
        assert result["final_output"] == "先休息一下。"
        assert result["_answer_delta_emitted"]
        assert len(calls) == 1 and closed == [True]
        events = db.execute(select(AgentEvent).order_by(AgentEvent.id)).scalars().all()
        assert not any(e.event_type in {"visible_thought", "visible_thought_delta"} for e in events)
        assert "".join(e.payload_json["text"] for e in events if e.event_type == "answer_delta") == "先休息一下。"


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks,reason", [(["workflow\nignored"], None), (["workflow"], None), (["oops\nsecret"], "invalid_header"),
    (["chat"], "incomplete_stream"), (["chat\n"], "incomplete_stream"), (["x" * 513], "header_too_long")])
async def test_workflow_and_protocol_fallback_emit_no_answer(env, monkeypatch, chunks, reason):
    calls, closed, _ = fake_model(monkeypatch, chunks)
    with env.factory() as db:
        result = await chat_entry(RuntimeNodes(db, {}), initial(env))
        assert result["chat_entry_route"] == "workflow"
        events = db.execute(select(AgentEvent)).scalars().all()
        assert not any(e.event_type.startswith("answer_") for e in events)
        fallback = [e for e in events if e.event_type == "chat_route_fallback"]
        assert len(fallback) == int(reason is not None)
        if reason:
            assert fallback[0].payload_json == {"reason": reason, "count": 1}
        assert len(calls) == 1 and closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], ["chat\n部分回复"]])
async def test_cancel_closes_actual_entry_stream(env, monkeypatch, chunks):
    calls, closed, ready = fake_model(monkeypatch, chunks, wait=True)
    token = execution.set(ChatExecution(env.run))
    try:
        with env.factory() as db:
            nodes = RuntimeNodes(db, {})
            async def entry(state):
                return await chat_entry(nodes, state)
            task = asyncio.create_task(controlled_node("chat_entry", entry, db)(initial(env)))
            await ready.wait()
            execution.get().cancelled = True
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            events = db.execute(select(AgentEvent)).scalars().all()
            assert any(e.event_type == "node_cancelled" for e in events)
            assert not any(e.event_type == "node_completed" for e in events)
            assert closed == [True]
    finally:
        execution.reset(token)


@pytest.mark.asyncio
async def test_provider_failure_is_not_protocol_retry(env, monkeypatch):
    calls, closed, _ = fake_model(monkeypatch, [], error=True)
    with env.factory() as db:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await chat_entry(RuntimeNodes(db, {}), initial(env))
        assert len(calls) == 1 and closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"attachment_ids": [1]}, {"route": "research"}, {"explicit_agent": "tool_agent"}, {"page_context": {"selected_feed_card_id": 1}}])
async def test_explicit_work_does_not_use_chat_model(env, monkeypatch, payload):
    calls, _, _ = fake_model(monkeypatch, ["chat\nwrong"])
    with env.factory() as db:
        result = await chat_entry(RuntimeNodes(db, payload), initial(env))
        assert result["chat_entry_route"] == "workflow"
        assert not calls


@pytest.mark.asyncio
async def test_latest_message_and_unfinished_context_reach_selected_model(env, monkeypatch):
    calls, _, _ = fake_model(monkeypatch, ["chat\n好的，只讲聊天。"])
    with env.factory() as db:
        nodes = RuntimeNodes(db, {"chat_continuation": "Assistant (unfinished): 原先在讲系统架构。"})
        state = {**initial(env), "user_input": "只讲聊天模块，简单一点"}
        await chat_entry(nodes, state)
    assert calls[0][0].type == "system"
    assert "Assistant (unfinished)" in calls[0][1].content
    assert calls[0][1].content.endswith("只讲聊天模块，简单一点")


@pytest.mark.asyncio
async def test_lifecycle_is_live_and_unique(env):
    from src.web_app.agent.runtime.checkpoint import record_step
    ready, release = asyncio.Event(), asyncio.Event()
    with env.factory() as db:
        async def work(state):
            ready.set()
            await release.wait()
            record_step(db, env.run, "work", "read", {}, {})
            return state
        task = asyncio.create_task(controlled_node("work", work, db)(initial(env)))
        await ready.wait()
        events = db.execute(select(AgentEvent)).scalars().all()
        assert [e.event_type for e in events] == ["node_started"]
        release.set()
        await task
        await controlled_node("work", work, db)(initial(env))
        events = db.execute(select(AgentEvent).order_by(AgentEvent.id)).scalars().all()
        starts = [e for e in events if e.event_type == "node_started"]
        ends = [e for e in events if e.event_type == "node_completed"]
        assert len(starts) == len(ends) == 2
        assert starts[0].payload_json["step_id"] != starts[1].payload_json["step_id"]


@pytest.mark.asyncio
async def test_cold_client_initialization_does_not_block_cancel(env, monkeypatch):
    from threading import Event
    entered, release = Event(), Event()
    def slow_factory(*args, **kwargs):
        entered.set()
        release.wait(3)
        return SimpleNamespace()
    monkeypatch.setattr("src.web_app.agent.runtime.chat_fast_path.get_chat_model", slow_factory)
    with env.factory() as db:
        task = asyncio.create_task(chat_entry(RuntimeNodes(db, {}), initial(env)))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, .3)
        finally:
            release.set()
