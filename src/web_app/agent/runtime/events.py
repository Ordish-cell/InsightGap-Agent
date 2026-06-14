import json
from datetime import datetime
from typing import Any, Iterable


def to_sse(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        event_name = event.get("event", "status")
        data = json.dumps(event.get("data", event), ensure_ascii=False, default=str)
        yield f"event: {event_name}\ndata: {data}\n\n"


# Channel constants for SSE events.
# Frontend renders based on display_channel, never on node_name.
DISPLAY_CHANNEL_THINKING = "thinking"
DISPLAY_CHANNEL_ANSWER = "answer"
DISPLAY_CHANNEL_TOOL = "tool"
DISPLAY_CHANNEL_STATUS = "status"

VISIBILITY_USER = "user"
VISIBILITY_TRACE = "trace"
VISIBILITY_INTERNAL = "internal"

# Map event_type → (visibility, display_channel).  Everything not listed
# defaults to ("trace", "status").
_EVENT_DISPLAY: dict[str, tuple[str, str]] = {
    "visible_thought_delta": (VISIBILITY_USER, DISPLAY_CHANNEL_THINKING),
    "visible_progress_delta": (VISIBILITY_USER, DISPLAY_CHANNEL_THINKING),
    "answer_started": (VISIBILITY_USER, DISPLAY_CHANNEL_ANSWER),
    "answer_delta": (VISIBILITY_USER, DISPLAY_CHANNEL_ANSWER),
    "answer_completed": (VISIBILITY_USER, DISPLAY_CHANNEL_ANSWER),
    "tool_call_started": (VISIBILITY_USER, DISPLAY_CHANNEL_TOOL),
    "tool_call_delta": (VISIBILITY_USER, DISPLAY_CHANNEL_TOOL),
    "tool_call_completed": (VISIBILITY_USER, DISPLAY_CHANNEL_TOOL),
    "tool_call_failed": (VISIBILITY_USER, DISPLAY_CHANNEL_TOOL),
    "approval_required": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "approval_granted": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "approval_rejected": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "run_created": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "run_completed": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "run_failed": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "run_paused": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "run_resumed": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "milestone_started": (VISIBILITY_USER, DISPLAY_CHANNEL_THINKING),
    "milestone_completed": (VISIBILITY_USER, DISPLAY_CHANNEL_THINKING),
}


def _event_channels(event_type: str) -> tuple[str, str]:
    """Return (visibility, display_channel) for an event type."""
    return _EVENT_DISPLAY.get(event_type, (VISIBILITY_TRACE, DISPLAY_CHANNEL_STATUS))


def queue_stream_event(
    queue: Any,
    event_type: str,
    payload: dict[str, Any],
    *,
    run_id: int | None = None,
    thread_id: str = "",
    node_name: str = "",
    visibility: str | None = None,
    display_channel: str | None = None,
) -> None:
    """Push an event to the SSE stream queue. Does NOT persist to DB.

    Callers that need DB persistence should call record_event() separately.

    The *visibility* and *display_channel* fields tell the frontend which
    UI region should consume this event.  When omitted they are resolved
    automatically from the event_type.
    """
    if queue is None:
        return
    resolved_visibility, resolved_channel = _event_channels(event_type)
    queue.put_nowait(
        {
            "event": event_type,
            "data": {
                "run_id": run_id,
                "thread_id": thread_id,
                "event_type": event_type,
                "node_name": node_name,
                "visibility": visibility or resolved_visibility,
                "display_channel": display_channel or resolved_channel,
                "payload": payload,
                "created_at": datetime.now().isoformat(),
            },
        }
    )
