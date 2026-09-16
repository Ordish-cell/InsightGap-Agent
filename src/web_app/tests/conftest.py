"""Shared pytest fixtures — Windows-compatible async event loop."""
from __future__ import annotations

import asyncio
import selectors
import sys

import pytest


@pytest.fixture
def selected_test_model():
    from src.web_app.agent.llm.context import ModelExecutionContext, use_model_context
    model = ModelExecutionContext(model_config_id=1, connection_id=1, connection_revision=1,
        provider="custom", protocol="openai_chat_completions", model="test-model", display_name="Test model",
        config={"base_url": "http://provider.invalid/v1"})
    with use_model_context(model):
        yield model


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use SelectorEventLoop on Windows for psycopg async compatibility."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def scripted_supervisor(monkeypatch):
    """Explicit action script for service tests; no external provider calls."""
    from types import SimpleNamespace
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.runtime.state import SupervisorAction
    from src.web_app.core.config import settings
    async def decide(self, state):
        if state.get("observations"):
            return SupervisorAction(action="respond")
        route = self.payload.get("route")
        if self.payload.get("tool_name"):
            return SupervisorAction(action="tool", arguments={"name": self.payload["tool_name"], "input": self.payload.get("tool_input", {})})
        if route == "artifact":
            return SupervisorAction(action="artifact", arguments={"title": "Test report", "content": "Test report content"})
        if route == "rag":
            return SupervisorAction(action="rag", arguments={"query": state["user_input"]})
        if route == "research":
            return SupervisorAction(action="deep_research", arguments={"query": state["user_input"]})
        if route == "skill":
            return SupervisorAction(action="skill", arguments={"operation": "match", "query": state["user_input"]})
        return SupervisorAction(action="respond")
    async def stream(prompt):
        yield SimpleNamespace(content="已完成测试回答。")
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: SimpleNamespace(astream=stream))
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.llm.native_turn import NativeTurnResult
    from src.web_app.tests.test_native_supervisor import call
    from langchain_core.messages import AIMessageChunk
    async def native_turn(self, state, *, tools_enabled=True):
        action = await decide(self, state) if tools_enabled else SupervisorAction(action="respond")
        if action.action == "respond":
            return NativeTurnResult("已完成测试回答。", AIMessageChunk(content="已完成测试回答。"), None), {}
        args, kind = action.arguments, action.action
        name = args["name"] if kind == "tool" else kind
        inputs = args.get("input", {}) if kind == "tool" else args
        message = call(name, inputs)
        return NativeTurnResult("", message, {**message.tool_calls[0], "name": name}), {name: kind}
    monkeypatch.setattr(SupervisorNodes, "model_turn", native_turn)
    from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter
    from src.web_app.research.schemas import ResearchResult
    async def research(self, **kwargs):
        return ResearchResult(summary="Test research evidence", markdown_report="# Test research", evidence=kwargs.get("evidence") or [], metadata={"engine": "test_provider"})
    monkeypatch.setattr(OpenDeepResearchAdapter, "run_research", research)
    yield
