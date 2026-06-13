import asyncio

from src.web_app.agent.runtime.node_groups.agent_nodes import _queue_tool_event
from src.web_app.services.agent_service import _tool_output_preview_for_frontend


def test_tool_stream_event_contains_frontend_trace_fields():
    queue = asyncio.Queue()
    state = {"run_id": 42, "thread_id": "thread-1", "_stream_queue": queue}

    _queue_tool_event(
        state,
        "tool_call_started",
        tool_call_id="tool-1",
        tool_name="local_file.read",
        args_preview={"path": "README.md", "api_key": "[redacted]"},
        status="running",
    )

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
            "content": "x" * 1000,
        },
        max_chars=120,
    )

    assert "secret-token" not in preview
    assert "_metadata" not in preview
    assert len(preview) <= 123
    assert preview.endswith("...")
