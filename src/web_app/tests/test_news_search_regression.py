"""News must reach the governed search runtime, including casual wording."""
import os
import pytest
from types import SimpleNamespace

from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_supervisor_loop import state

@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    '{"action":"tool","arguments":{"name":"web.search"}}',
    '```json\n{"action":"tool","arguments":{"name":"web.search"}}\n```',
    '[{"action":"respond"},{"action":"respond"}]',
    '{"action":"tool","skip_approval":true}',
])
async def test_textual_actions_are_never_executed(text):
    from src.web_app.agent.llm.native_turn import collect_native_turn
    from src.web_app.tests.test_native_supervisor import Model
    from langchain_core.messages import AIMessageChunk, HumanMessage
    result = await collect_native_turn(Model([[AIMessageChunk(content=text)]]), [HumanMessage(content="hello")], [], lambda delta: None)
    assert result.tool_call is None and result.text == text


@pytest.mark.asyncio
async def test_news_entry_search_answer_contract(env, monkeypatch):
    from src.web_app.agent.runtime.graph import AgentRuntime
    from src.web_app.services.mcp_service import mcp_service
    from src.web_app.core.config import settings
    calls = []
    from src.web_app.tests.test_native_supervisor import Model, call
    from langchain_core.messages import AIMessageChunk
    model = Model([[call("web.search", {"query": "today news"})], [AIMessageChunk(content="News summary [E1]")]])
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", False)
    def search(*a, **kw):
        calls.append(kw["tool_name"])
        return {"id": 999, "tool_name": "web.search", "status": "completed", "output": {"results": [{"title": "News evidence", "snippet": "News evidence", "url": "https://example.com/news"}]}}
    monkeypatch.setattr(mcp_service, "call_tool", search)
    with env.factory() as db:
        result = await AgentRuntime(db, {}).run({**state(env), "user_input": "今天有啥新闻？"})
    assert calls == ["web.search"]
    assert result["status"] == "completed"
    assert result["observations"][0]["status"] == "ok"
    assert "[E1]" in result["final_answer"]


@pytest.mark.asyncio
@pytest.mark.skipif(os.environ.get("NEWS_LIVE_SMOKE") != "1", reason="Opt-in real selected model and web search; isolated application database")
async def test_real_casual_news_uses_search(env, monkeypatch):
    from src.web_app.db.session import SessionLocal
    from src.web_app.models.orm import AgentRun
    from src.web_app.services.llm_registry_service import resolve_run_model_context
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.agent.runtime.graph import AgentRuntime
    from src.web_app.core.config import settings
    from src.web_app.services.memory_service import memory_service
    with SessionLocal() as source:
        original = source.get(AgentRun, int(os.environ["NEWS_SMOKE_SOURCE_RUN"]))
        assert original is not None
        context = resolve_run_model_context(source, original.user_id, original.graph_state["model_context"])
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", False)
    monkeypatch.setattr(settings, "agent_max_supervisor_steps", 4)
    monkeypatch.setattr(memory_service, "get_baseline_memories", lambda *a, **k: [])
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    with use_model_context(context), env.factory() as db:
        result = await AgentRuntime(db, {}).run({**state(env), "user_input": "今天有啥新闻？", "model_context": context.public_dict()})
    searches = [r for r in result.get("tool_calls", []) if r["tool_name"] == "web.search"]
    assert searches, {k: result.get(k) for k in ("chat_entry_route", "current_action", "errors", "final_answer")}
    assert any(r["output"].get("results") for r in searches), "Search returned no evidence"
    assert result["status"] == "completed", {k: result.get(k) for k in ("errors", "error", "final_answer", "termination_reason")}
    assert result.get("observations")
    print("NEWS_LIVE_RESULT", {"model": context.model, "searches": len(searches), "evidence": sum(len(r.get("evidence", [])) for r in result["observations"]), "status": result["status"]})
