"""Real isolated graph contracts for the unified native Supervisor."""
import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.web_app.agent.llm.native_turn import tool_alias
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.agent.runtime.graph_builder import build_graph
from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_supervisor_loop import state
from src.web_app.tests.test_conversation_document_chat import add_file


class Model:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []
        self.tools = []

    def bind_tools(self, tools):
        from copy import copy
        bound = copy(self)
        bound.tools = tools
        return bound

    async def astream(self, messages):
        self.requests.append(messages)
        for chunk in next(self.turns):
            yield chunk


def call(name, args, text=""):
    return AIMessageChunk(content=text, tool_calls=[{"id": "provider-call-1", "name": tool_alias(name), "args": args}])


def configure(nodes, model, monkeypatch):
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: model)
    async def bootstrap(s):
        s.update(loop_protocol_version=1, context={})
        s.setdefault("save_policy", {})
        return s
    monkeypatch.setattr(nodes, "bootstrap_context", bootstrap)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [("processing", "empty"), ("failed", "failed")])
async def test_unready_document_observation_preserves_status(env, status, expected):
    doc = add_file(env, status=status)
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = state(env)
        await nodes.permission_guard(s)
        s["current_action"] = {"action": "document_read", "action_id": "read", "arguments": {"document_ids": [doc]}}
        await nodes.document_read(s)
    result = s["observations"][0]
    assert result["status"] == expected and not result["evidence"]
    assert result["data"]["reads"][0]["status"] == status
    assert status in result["warnings"][0]


@pytest.mark.asyncio
async def test_required_context_overflow_stops_before_provider(env, monkeypatch):
    model = Model([])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, model, monkeypatch)
        monkeypatch.setattr(nodes, "_context_limit", lambda: 1)
        result = await build_graph(nodes).ainvoke(state(env))
    assert result["status"] == "failed" and result["error"] == "context_budget_exhausted"
    assert not model.requests


@pytest.mark.asyncio
async def test_direct_answer_one_model_call(env, monkeypatch):
    model = Model([[AIMessageChunk(content="Hello")]])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, model, monkeypatch)
        result = await build_graph(nodes).ainvoke(state(env))
    assert result["final_answer"] == "Hello"
    assert result["status"] == "completed" and len(model.requests) == 1
    assert "chat_entry_route" not in result


@pytest.mark.asyncio
async def test_read_then_search_then_answer_with_native_continuation(env, monkeypatch):
    from src.web_app.agent.runtime.capabilities import observe
    from src.web_app.agent.runtime.state import CapabilityResult
    doc = add_file(env, text="Document unique fact")
    model = Model([
        [call("document.read", {"document_ids": [doc]}, "Reading the attachment.")],
        [call("web.search", {"query": "verify current fact"})],
        [AIMessageChunk(content="Verified [E1] [E2]")],
    ])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {"attachment_ids": [doc]})
        configure(nodes, model, monkeypatch)
        async def search(s):
            assert s["current_action"]["arguments"]["name"] == "web.search"
            return observe(nodes, s, CapabilityResult(action_id=s["current_action"]["action_id"], capability="tool", status="ok", summary="Search fact", evidence=[{"quote": "Search fact"}]))
        monkeypatch.setattr(nodes, "tool_runtime", search)
        result = await build_graph(nodes).ainvoke(state(env))
    assert len(model.requests) == 3 and result["status"] == "completed"
    assert [r["capability"] for r in result["observations"]] == ["document_read", "tool"]
    assert any(isinstance(m, ToolMessage) and "Document unique fact" in m.content for m in model.requests[1])
    assert result["runtime_budget"]["tool_calls"] == 1


@pytest.mark.asyncio
async def test_tool_budget_blocks_before_execution(env, monkeypatch):
    model = Model([[call("web.search", {"query": "news"})], [AIMessageChunk(content="Existing results only")]])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, model, monkeypatch)
        initial = {**state(env), "runtime_budget": {"steps": 0, "tool_calls": 8, "deep_research_calls": 0, "consecutive_failures": 0}}
        result = await build_graph(nodes).ainvoke(initial)
    assert result["termination_reason"] == "tool_budget_exhausted"
    assert result["runtime_budget"]["tool_calls"] == 8
    assert result["observations"][0]["status"] == "blocked"


@pytest.mark.asyncio
async def test_multiple_calls_execute_nothing_then_repair(env, monkeypatch):
    doc = add_file(env)
    bad = AIMessageChunk(content="", tool_calls=[
        {"id": str(i), "name": tool_alias("web.search"), "args": {"query": "q"}} for i in range(2)])
    model = Model([[call("document.read", {"document_ids": [doc]})], [bad], [AIMessageChunk(content="No search tools executed")]])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, model, monkeypatch)
        result = await build_graph(nodes).ainvoke(state(env))
    assert result["status"] == "completed" and result["runtime_budget"]["tool_calls"] == 0
    assert result["observations"][-1]["error"] == "multiple_tool_calls_not_allowed"
    assert any(isinstance(m, ToolMessage) for m in model.requests[1])
    assert not any(isinstance(m, ToolMessage) for m in model.requests[2])
    assert "Invalid single-action JSON" not in str(model.requests[2])


@pytest.mark.asyncio
async def test_native_text_event_classification(env, monkeypatch):
    from sqlalchemy import select
    from src.web_app.models.orm import AgentEvent
    doc = add_file(env)
    model = Model([[call("document.read", {"document_ids": [doc]}, "Checking")], [AIMessageChunk(content="Answer")]])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, model, monkeypatch)
        await build_graph(nodes).ainvoke(state(env))
        events = list(db.scalars(select(AgentEvent).where(AgentEvent.run_id == env.run).order_by(AgentEvent.id)))
    completed = [e.payload_json for e in events if e.event_type == "agent_text_completed"]
    assert [e["role"] for e in completed] == ["progress", "final"]
    assert len({e["text_id"] for e in completed}) == 2
    assert len([e for e in events if e.event_type == "answer_completed"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approved", "rejected"])
async def test_real_interrupt_rebuild_does_not_reselect_action(env, monkeypatch, decision):
    from langgraph.types import Command
    from src.web_app.db.repositories.approval_repository import ApprovalRepository
    from src.web_app.mcp.local_provider import local_provider
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": f"run:{env.run}"}}
    executed = []
    monkeypatch.setattr(local_provider, "call", lambda *a, **k: executed.append(1) or {"success": True})
    model = Model([[call("email.send", {"to": "test@example.com", "subject": "test", "body": "hello"}, "Preparing the requested message.")]])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, model, monkeypatch)
        paused = await build_graph(nodes, saver).ainvoke(state(env), config)
        payload = paused["__interrupt__"][0].value
        assert not executed
        approval = ApprovalRepository(db).get_by_user(env.user, payload["approval_id"])
        ApprovalRepository(db).decide_pending(approval, decision, approval.payload)
    resumed_model = Model([[AIMessageChunk(content="Done")]])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, resumed_model, monkeypatch)
        result = await build_graph(nodes, saver).ainvoke(Command(resume={"action": decision, "approval_id": payload["approval_id"], "tool_call_id": payload["tool_call_id"]}), config)
    assert result["status"] == "completed"
    assert len(model.requests) == len(resumed_model.requests) == 1
    assert len(executed) == (1 if decision == "approved" else 0)
    assert result["runtime_budget"]["tool_calls"] == 1


@pytest.mark.asyncio
async def test_recover_partial_native_text_never_restarts_model(env, monkeypatch):
    from src.web_app.agent.runtime.finalization import emit
    model = Model([])
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, model, monkeypatch)
        s = state(env)
        emit(nodes, s, "agent_text_started", {"text_id": "interrupted-turn", "model_turn_id": "interrupted-turn"})
        emit(nodes, s, "agent_text_delta", {"text_id": "interrupted-turn", "text": "Visible partial"})
        result = await build_graph(nodes).ainvoke(s)
    assert result["status"] == "failed" and result["final_answer"] == "Visible partial"
    assert not model.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["chat", "document", "news"])
async def test_live_native_graph(env, monkeypatch, scenario):
    import os
    if os.environ.get("NATIVE_LIVE_SMOKE") != "1":
        pytest.skip("Opt-in native model and search provider; isolated application database")
    from src.web_app.db.session import SessionLocal
    from src.web_app.services.llm_registry_service import resolve_model_context
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.models.orm import DocumentChunk
    from sqlalchemy import select
    with SessionLocal() as source:
        user_id = int(os.environ["NATIVE_TEST_USER_ID"])
        context = resolve_model_context(source, user_id, int(os.environ["NATIVE_MODEL_CONFIG_ID"]))
        chunks = list(source.scalars(select(DocumentChunk).where(DocumentChunk.document_id == int(os.environ["NATIVE_TEST_DOCUMENT_ID"]), DocumentChunk.user_id == user_id))) if scenario == "document" else []
        body = "\n".join(c.content for c in chunks if (c.metadata_json or {}).get("chunk_role") == "parent")
    doc = add_file(env, name="daily.md", text=body) if scenario == "document" else None
    request = {"attachment_ids": [doc]} if doc else {}
    prompts = {"chat": "Please say hello in five words.", "document": "Summarize this attached document. Read it first.", "news": "Search the web for today's technology news and summarize with sources."}
    from src.web_app.agent.runtime.graph import AgentRuntime
    from src.web_app.core.config import settings
    from src.web_app.services.rag_service import rag_service
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", False)
    # Never query development vectors from this isolated document fixture.
    monkeypatch.setattr(rag_service, "ask", lambda *a, **k: {"retrieval_status": "failed", "evidence": [], "error": "Use document.read in isolated smoke"})
    with env.factory() as db, use_model_context(context):
        result = await AgentRuntime(db, request).run({**state(env), "user_input": prompts[scenario], "model_context": context.public_dict()})
    print("NATIVE_LIVE_TRACE", {"scenario": scenario, "steps": result["runtime_budget"]["steps"], "error": result.get("error"), "observations": [{"capability": r["capability"], "status": r["status"], "error": r["error"]} for r in result.get("observations", [])]})
    assert result["status"] == "completed", (result.get("error"), result.get("native_protocol_error"))
    if scenario == "chat":
        assert result["runtime_budget"]["steps"] == 1
    elif scenario == "document":
        assert any(r["capability"] in {"document_read", "rag"} and r["status"] == "ok" for r in result["observations"])
    else:
        assert any(c["tool_name"] == "web.search" and c["status"] == "completed" for c in result.get("tool_calls", []))
    print("NATIVE_LIVE", {"scenario": scenario, "model": context.model, "steps": result["runtime_budget"]["steps"], "status": result["status"]})
