import asyncio
from threading import Event

import pytest
from sqlalchemy import select

from src.web_app.agent.llm.context import ModelExecutionContext, get_model_context, use_model_context
from src.web_app.agent.runtime.chat_control import ChatExecution, execution
from src.web_app.models.orm import AgentChatMessage, AgentConversationSummary
from src.web_app.services.summary_tasks import SummaryTasks, update_pending
from src.web_app.tests.test_chat_control import env


@pytest.mark.asyncio
async def test_slow_summary_coalesces_and_does_not_block_loop(monkeypatch):
    manager = SummaryTasks()
    started, release = Event(), Event()
    calls = []
    def slow(*key):
        assert execution.get() is None
        calls.append((key, get_model_context().model))
        started.set()
        release.wait(3)
    monkeypatch.setattr("src.web_app.services.summary_tasks.update_pending", slow)
    monkeypatch.setattr("src.web_app.services.summary_tasks.settings.enable_conversation_summary", True)
    ctx = ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "selected", "Selected")
    token = execution.set(ChatExecution(1, cancelled=True))
    try:
        with use_model_context(ctx):
            manager.schedule(1, "chat")
            while not started.is_set():
                await asyncio.sleep(.001)
            manager.schedule(1, "chat")
            manager.schedule(1, "chat")
            assert len(calls) == 1  # second update cannot race the first
            release.set()
            await asyncio.wait_for(asyncio.gather(*tuple(manager.tasks.values())), 3)
        assert len(calls) == 2
        assert all(model == "selected" for _, model in calls)
    finally:
        release.set()
        execution.reset(token)
        await manager.shutdown()


def test_summary_cursor_is_idempotent_and_skips_interrupted(env, monkeypatch):
    import src.web_app.services.conversation_summary_service as module
    from src.web_app.services.conversation_summary_service import conversation_summary_service as service
    calls = []
    monkeypatch.setattr("src.web_app.services.summary_tasks.SessionLocal", env.factory)
    monkeypatch.setattr("src.web_app.services.summary_tasks.settings.enable_conversation_summary", True)
    monkeypatch.setattr(module, "_llm_call", lambda prompt: calls.append(prompt) or '{"summary_text":"saved"}')
    monkeypatch.setattr(service, "create_segment_if_needed", lambda **kwargs: [])
    with env.factory() as db:
        msg = db.scalar(select(AgentChatMessage).where(AgentChatMessage.role == "assistant"))
        msg.content, msg.status = "partial secret conclusion", "interrupted"
        db.commit()
    update_pending(env.user, "chat")
    assert not calls
    with env.factory() as db:
        msg = db.scalar(select(AgentChatMessage).where(AgentChatMessage.role == "assistant"))
        msg.content, msg.status = "finished", "completed"
        cursor = msg.id
        db.commit()
    update_pending(env.user, "chat")
    update_pending(env.user, "chat")
    assert len(calls) == 1
    with env.factory() as db:
        row = db.scalar(select(AgentConversationSummary))
        assert row.last_message_id == cursor
