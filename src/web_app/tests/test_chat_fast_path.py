import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src.web_app.agent.runtime.chat_control import ChatExecution, controlled_node, execution
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.models.orm import AgentEvent
from src.web_app.tests.test_chat_control import env






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
    from langchain_core.messages import AIMessageChunk
    from src.web_app.agent.llm.content import message_text
    class NativeModel:
        def bind_tools(self, tools):
            return self
        async def astream(self, prompt):
            async for chunk in stream(prompt):
                text = message_text(chunk)
                if text:
                    yield AIMessageChunk(content=text)
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: NativeModel())
    return calls, closed, ready


def initial(env):
    return {"run_id": env.run, "user_id": env.user, "conversation_id": "chat", "thread_id": "chat-thread",
            "user_input": "唉，我好累", "interaction_version": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_version", [2])
@pytest.mark.parametrize("question", ["唉，我好累", "你好", "解释一下哈希表"])
async def test_compiled_graph_chat_is_one_call(env, monkeypatch, question, runtime_version):
    from src.web_app.agent.runtime.graph import AgentRuntime
    calls, closed, _ = fake_model(monkeypatch, [[{"type": "reasoning", "text": "private"}], "先休息", "一下。"])
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    with env.factory() as db:
        runtime = AgentRuntime(db, {"runtime_version": runtime_version})
        async def forbidden(state):
            pytest.fail("Chat entered heavy workflow")
        for name in ("capability", "tool_runtime", "document_read", "deep_research"):
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
@pytest.mark.parametrize("chunks", [[], ["部分回复"]])
async def test_cancel_closes_actual_entry_stream(env, monkeypatch, chunks):
    calls, closed, ready = fake_model(monkeypatch, chunks, wait=True)
    token = execution.set(ChatExecution(env.run))
    try:
        with env.factory() as db:
            nodes = SupervisorNodes(db, {})
            async def entry(state):
                return await run_native(nodes, state)
            task = asyncio.create_task(controlled_node("supervisor", entry, db)(initial(env)))
            await ready.wait()
            execution.get().cancelled = True
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            events = db.execute(select(AgentEvent)).scalars().all()
            assert any(e.event_type == "node_cancelled" for e in events)
            assert not any(e.event_type == "node_completed" and e.node_name == "supervisor" for e in events)
            assert closed == [True]
    finally:
        execution.reset(token)


@pytest.mark.asyncio
async def test_provider_failure_is_not_protocol_retry(env, monkeypatch):
    calls, closed, _ = fake_model(monkeypatch, [], error=True)
    with env.factory() as db:
        result = await run_native(SupervisorNodes(db, {}), initial(env))
        assert result["status"] == "failed" and result["error"] == "supervisor_unavailable"
        assert len(calls) == 1 and closed == [True]




@pytest.mark.asyncio
async def test_latest_message_and_unfinished_context_reach_selected_model(env, monkeypatch):
    calls, _, _ = fake_model(monkeypatch, ["好的，只讲聊天。"])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {"chat_continuation": "Assistant (unfinished): 原先在讲系统架构。"})
        state = {**initial(env), "user_input": "只讲聊天模块，简单一点"}
        await run_native(nodes, state)
    assert calls[0][0].type == "system"
    assert "Assistant (unfinished)" in calls[0][1].content
    assert 0 <= calls[0][1].content.find("只讲聊天模块，简单一点")


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
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", slow_factory)
    with env.factory() as db:
        task = asyncio.create_task(run_native(SupervisorNodes(db, {}), initial(env)))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, .3)
        finally:
            release.set()


async def run_native(nodes, state):
    from src.web_app.agent.runtime.graph_builder import build_graph
    return await build_graph(nodes).ainvoke(state)
