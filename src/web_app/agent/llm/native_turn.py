"""Native tool protocol adapter; no routing, execution, or text-to-tool fallback.

Callers own event persistence and must retain visible partial text
when this adapter raises after calling on_text.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import AIMessageChunk

from .content import message_text


class NativeProtocolError(RuntimeError):
    """Stable public code; never include provider credentials or response bodies."""


@dataclass(frozen=True)
class NativeTurnResult:
    text: str
    message: AIMessageChunk
    tool_call: dict[str, Any] | None

    @property
    def text_role(self) -> str:
        return "progress" if self.tool_call else "final"


def tool_alias(name: str) -> str:
    """Stable ASCII provider name; digest distinguishes punctuation collisions."""
    readable = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:40]
    return "ig_" + readable + "_" + hashlib.sha256(name.encode()).hexdigest()[:16]


def compile_tools(catalog: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    definitions, names = [], {}
    for spec in catalog:
        name = spec["name"]
        alias = tool_alias(name)
        if alias in names:
            raise NativeProtocolError("duplicate_tool_name")
        names[alias] = name
        definitions.append({
            "name": alias,
            "description": spec.get("description", ""),
            "parameters": spec["input_schema"],
        })
    return definitions, names


async def collect_native_turn(
    model: Any,
    messages: list[Any],
    catalog: list[dict[str, Any]],
    on_text: Callable[[str], Any],
    *,
    timeout_seconds: float = 30,
) -> NativeTurnResult:
    """Stream tentative text immediately; classify only after the full turn.

    The returned tool call is validated structurally, never executed here. Its
    original provider id and aliased name remain in message for ToolMessage
    continuation; tool_call.name resolves to the internal business name.
    """
    definitions, names = compile_tools(catalog)
    try:
        options = {}
        if definitions and "parallel_tool_calls" in inspect.signature(model.bind_tools).parameters:
            options["parallel_tool_calls"] = False
        bound = model.bind_tools(definitions, **options) if definitions else model
    except (AttributeError, NotImplementedError) as exc:
        raise NativeProtocolError("model_tool_calling_unsupported") from exc

    response = None
    text = ""
    try:
        async with asyncio.timeout(timeout_seconds):
            stream = bound.astream(messages)
            try:
                async for chunk in stream:
                    if not isinstance(chunk, AIMessageChunk):
                        raise NativeProtocolError("unexpected_model_chunk")
                    response = chunk if response is None else response + chunk
                    delta = message_text(chunk)
                    if delta:
                        text += delta
                        pending = on_text(delta)
                        if inspect.isawaitable(pending):
                            await pending
            finally:
                await stream.aclose()
    except TimeoutError as exc:
        raise NativeProtocolError("model_stream_timeout") from exc
    except ValueError as exc:
        # LangChain raises before yielding when the provider closes an empty
        # stream. Treat only this exact, output-free case as an empty response.
        if response is None and str(exc) == "No generation chunks were returned":
            raise NativeProtocolError("model_stream_empty") from exc
        raise

    if response is None:
        raise NativeProtocolError("model_stream_empty")
    if response.invalid_tool_calls:
        raise NativeProtocolError("invalid_tool_arguments")
    calls = response.tool_calls
    if len(calls) > 1:
        raise NativeProtocolError("multiple_tool_calls_not_allowed")
    call = None
    if calls:
        original = calls[0]
        if not original.get("id"):
            raise NativeProtocolError("tool_call_id_missing")
        if original.get("name") not in names:
            raise NativeProtocolError("unknown_tool_name")
        if not isinstance(original.get("args"), dict):
            raise NativeProtocolError("invalid_tool_arguments")
        call = {**original, "name": names[original["name"]]}
    elif not text.strip():
        raise NativeProtocolError("model_response_empty")
    return NativeTurnResult(text=text, message=response, tool_call=call)
