"""Provider-free protocol tests for the unified Supervisor migration gate."""
import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from src.web_app.agent.llm.native_turn import (
    NativeProtocolError, collect_native_turn, compile_tools, tool_alias,
)

CATALOG = [{"name": "web.search", "description": "Search", "input_schema": {
    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
}}]


class Model:
    def __init__(self, chunks):
        self.chunks, self.bound, self.closed = chunks, None, False

    def bind_tools(self, definitions):
        self.bound = definitions
        return self

    async def astream(self, messages):
        try:
            for chunk in self.chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        finally:
            self.closed = True


def call_chunk(name=None, args='{"query":"news"}', id="call-1", index=0):
    return AIMessageChunk(content="", tool_call_chunks=[{
        "name": name or tool_alias("web.search"), "args": args, "id": id, "index": index,
    }])


@pytest.mark.asyncio
async def test_direct_answer_is_one_stream_without_tools():
    model, visible = Model([AIMessageChunk(content="Hello"), AIMessageChunk(content=" world")]), []
    result = await collect_native_turn(model, [], CATALOG, visible.append)
    assert visible == ["Hello", " world"]
    assert result.text == "Hello world" and result.text_role == "final"
    assert result.tool_call is None and model.closed
    assert model.bound[0]["name"] == tool_alias("web.search")


@pytest.mark.asyncio
async def test_fragmented_call_keeps_id_and_classifies_preamble():
    model = Model([
        AIMessageChunk(content="Checking sources."),
        call_chunk(args='{"query":'),
        AIMessageChunk(content="", tool_call_chunks=[{"index": 0, "name": None, "id": None, "args": '"news"}'}]),
    ])
    visible = []
    result = await collect_native_turn(model, [], CATALOG, visible.append)
    assert visible == ["Checking sources."] and result.text_role == "progress"
    assert result.tool_call["name"] == "web.search"
    assert result.tool_call["args"] == {"query": "news"}
    assert result.tool_call["id"] == "call-1"
    assert result.message.tool_calls[0]["name"] == tool_alias("web.search")


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks,code", [
    ([], "model_stream_empty"),
    ([AIMessageChunk(content="")], "model_response_empty"),
    ([call_chunk(name="invented")], "unknown_tool_name"),
    ([call_chunk(id=None)], "tool_call_id_missing"),
    ([call_chunk(args="not json")], "invalid_tool_arguments"),
    ([call_chunk(), call_chunk(id="call-2", index=1)], "multiple_tool_calls_not_allowed"),
])
async def test_invalid_protocol_never_returns_executable_call(chunks, code):
    with pytest.raises(NativeProtocolError, match=code):
        await collect_native_turn(Model(chunks), [], CATALOG, lambda text: None)


@pytest.mark.asyncio
async def test_no_text_json_fallback():
    text = '{"action":"tool","arguments":{"name":"web.search","input":{}}}'
    result = await collect_native_turn(Model([AIMessageChunk(content=text)]), [], CATALOG, lambda text: None)
    assert result.tool_call is None


@pytest.mark.asyncio
async def test_partial_stream_failure_preserves_visible_text_and_closes():
    visible, model = [], Model([AIMessageChunk(content="Partial"), RuntimeError("provider failed")])
    with pytest.raises(RuntimeError, match="provider failed"):
        await collect_native_turn(model, [], CATALOG, visible.append)
    assert visible == ["Partial"] and model.closed


@pytest.mark.asyncio
async def test_reasoning_is_not_visible_answer():
    chunk = AIMessageChunk(content=[{"type": "reasoning", "summary": []}, {"type": "text", "text": "Answer"}])
    visible = []
    result = await collect_native_turn(Model([chunk]), [], CATALOG, visible.append)
    assert visible == ["Answer"] and result.text == "Answer"


@pytest.mark.asyncio
async def test_missing_native_binding_fails_explicitly():
    with pytest.raises(NativeProtocolError, match="model_tool_calling_unsupported"):
        await collect_native_turn(object(), [], CATALOG, lambda text: None)


@pytest.mark.asyncio
async def test_deadline_closes_stream():
    class Slow(Model):
        async def astream(self, messages):
            try:
                yield AIMessageChunk(content="Start")
                await asyncio.sleep(10)
            finally:
                self.closed = True
    model = Slow([])
    with pytest.raises(NativeProtocolError, match="model_stream_timeout"):
        await collect_native_turn(model, [], CATALOG, lambda text: None, timeout_seconds=0.01)
    assert model.closed


def test_aliases_are_stable_bounded_and_unique():
    assert tool_alias("web.search") != tool_alias("web_search")
    assert tool_alias("web.search") == tool_alias("web.search")
    assert len(tool_alias("x" * 200)) <= 64
    with pytest.raises(NativeProtocolError, match="duplicate_tool_name"):
        compile_tools(CATALOG + CATALOG)
