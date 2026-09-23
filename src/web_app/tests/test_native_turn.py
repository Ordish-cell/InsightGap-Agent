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
async def test_langchain_empty_stream_is_classified_but_other_value_errors_are_not():
    with pytest.raises(NativeProtocolError, match="model_stream_empty"):
        await collect_native_turn(Model([ValueError("No generation chunks were returned")]), [], CATALOG, lambda _: None)
    for chunks in ([ValueError("unrelated")], [AIMessageChunk(content="Partial"), ValueError("No generation chunks were returned")]):
        with pytest.raises(ValueError):
            await collect_native_turn(Model(chunks), [], CATALOG, lambda _: None)


def test_error_diagnostics_keep_safe_cause_and_stack_without_provider_body():
    from src.web_app.agent.llm.diagnostics import error_diagnostics
    try:
        try:
            raise ValueError("No generation chunks were returned")
        except ValueError as exc:
            raise NativeProtocolError("model_stream_empty") from exc
    except NativeProtocolError as exc:
        diagnostic = error_diagnostics(exc)["error_diagnostic"]
    assert "model_stream_empty" in diagnostic and "No generation chunks were returned" in diagnostic
    assert "test_native_turn.py" in diagnostic
    assert "secret-token" not in str(error_diagnostics(ValueError('Authorization: secret-token; private prompt')))


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_responses_failure_on_chat_completions_is_not_lost_by_sdk(partial):
    import httpx
    import json
    from src.web_app.agent.llm.stream_model import NativeStreamChatOpenAI
    from src.web_app.agent.llm.errors import ProviderStreamError
    from src.web_app.agent.llm.diagnostics import error_diagnostics
    chunks = []
    if partial:
        chunks.append({"id": "c", "object": "chat.completion.chunk", "model": "isolated", "created": 1,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Partial"}, "finish_reason": None}]})
    chunks.append({"type": "response.failed", "response": {"status": "failed", "error": {
        "code": "server_error", "message": "<400> InternalError.Algo.DataInspectionFailed: secret-token private-input"}}})
    body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    def handle(request):
        assert json.loads(request.content)["parallel_tool_calls"] is False
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = NativeStreamChatOpenAI(model="isolated", api_key="isolated", http_async_client=client, max_retries=0)
        visible = []
        with pytest.raises(ProviderStreamError, match="model_input_rejected") as caught:
            await collect_native_turn(model, [], CATALOG, visible.append)
    assert visible == (["Partial"] if partial else [])
    assert "secret-token" not in str(error_diagnostics(caught.value))


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
