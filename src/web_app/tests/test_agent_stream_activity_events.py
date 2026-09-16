import asyncio
import pytest

from src.web_app.tests.test_chat_control import env  # noqa: F401

from types import SimpleNamespace
from src.web_app.agent.runtime.finalization import emit
from src.web_app.services.agent_service import _tool_output_preview_for_frontend

@pytest.mark.asyncio
async def test_actual_tool_runtime_emits_redacted_previews(env, monkeypatch):
    from src.web_app.agent.runtime.tools import execute_tool
    from src.web_app.services.mcp_service import mcp_service
    monkeypatch.setattr(mcp_service, "get_tool", lambda *a: {"enabled": True, "safety_level": "L0_READ_ONLY", "input_schema": {}})
    monkeypatch.setattr(mcp_service, "call_tool", lambda *a, **k: {"id": 999, "tool_name": "preview.test", "status": "completed", "output": {"summary": "Visible result", "access_token": "never-publish", "reasoning": "private-trace"}})
    queue = asyncio.Queue()
    state = {"run_id": env.run, "user_id": env.user, "user_input": "preview", "current_action": {"action": "tool", "action_id": "preview-action"}}
    with env.factory() as db:
        result = await execute_tool(SimpleNamespace(db=db, _stream_queue=queue), state, "preview.test", {"query": "topic", "password": "never-publish"})
    assert result.status == "ok"
    payloads = []
    while not queue.empty():
        payloads.append(queue.get_nowait()["data"]["payload"])
    assert payloads[0]["argsPreview"]["query"] == "topic"
    assert "Visible result" in payloads[-1]["outputPreview"]
    assert "never-publish" not in str(payloads)
    assert "private-trace" not in str(payloads)


def test_tool_stream_event_contains_frontend_trace_fields(env):
    queue = asyncio.Queue()
    state = {"user_id": env.user, "run_id": env.run, "thread_id": "thread-1", "_stream_queue": queue}

    with env.factory() as db:
        emit(SimpleNamespace(db=db, _stream_queue=queue), state, "tool_call_started", {
            "tool_call_id": "tool-1", "tool_name": "local_file.read",
            "args_preview": {"path": "README.md", "api_key": "[redacted]"}, "status": "running",
        })


    item = queue.get_nowait()
    assert item["event"] == "tool_call_started"
    data = item["data"]
    assert data["event_type"] == "tool_call_started"
    assert data["display_channel"] == "tool"
    payload = data["payload"]
    assert payload["toolCallId"] == "tool-1"
    assert payload["tool_call_id"] == "tool-1"
    assert payload["toolName"] == "local_file.read"
    assert payload["argsPreview"]["path"] == "README.md"


def test_tool_output_preview_redacts_sensitive_fields_and_truncates():
    preview = _tool_output_preview_for_frontend(
        {
            "status": "completed",
            "token": "secret-token",
            "_metadata": {"approval_id": 1},
            "output": {"_metadata": {"approval_id": 2}, "nested_token": "nested-secret"},
            "content": "x" * 1000,
        },
        max_chars=120,
    )

    assert "secret-token" not in preview
    assert "nested-secret" not in preview
    assert "_metadata" not in preview
    assert len(preview) <= 123
    assert preview.endswith("...")
