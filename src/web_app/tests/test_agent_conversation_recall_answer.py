"""Recent conversation recall remains available without semantic retrieval or writes."""
from types import SimpleNamespace
from langchain_core.messages import AIMessageChunk
import pytest
from src.web_app.models.orm import AgentChatMessage
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.services.memory_service import memory_service
from src.web_app.tests.test_chat_control import env as env
from src.web_app.tests.test_supervisor_loop import state

@pytest.mark.asyncio
async def test_recent_history_reaches_answer_without_semantic_search(env, monkeypatch):
    monkeypatch.setattr(memory_service, "search_memory", lambda *a, **k: pytest.fail("unrequested semantic recall"))
    prompts = []
    async def stream(prompt):
        prompts.append(prompt)
        yield AIMessageChunk(content="你问过项目架构和手动测试。")
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: SimpleNamespace(astream=stream, bind_tools=lambda tools: SimpleNamespace(astream=stream)))
    with env.factory() as db:
        for index, content in enumerate(["项目架构是什么？", "怎么手动测试？"]):
            db.add(AgentChatMessage(message_id=f"recall-{index}", user_id=env.user,
                conversation_id="chat", role="user", content=content, status="completed"))
        db.commit()
        nodes = SupervisorNodes(db, {"chat_continuation": "只说重点"})
        s = state(env)
        s["user_input"] = "我前面问过什么？"
        await nodes.permission_guard(s)
        await nodes.bootstrap_context(s)
        await nodes.supervisor(s)
    assert len(prompts) == 1
    for text in ("项目架构是什么", "怎么手动测试", "只说重点"):
        assert text in prompts[0][1].content
    assert s["final_answer"] == "你问过项目架构和手动测试。"
    assert not s.get("memory_updates")
