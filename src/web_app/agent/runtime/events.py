import json
from datetime import datetime
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


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
    "run_interrupted": (VISIBILITY_USER, DISPLAY_CHANNEL_STATUS),
    "milestone_started": (VISIBILITY_USER, DISPLAY_CHANNEL_THINKING),
    "milestone_completed": (VISIBILITY_USER, DISPLAY_CHANNEL_THINKING),
}


class AgentEventPayload(BaseModel):
    """Validated payload for user-visible protocol events.

    Event-specific fields remain forward compatible while the common wire
    contract rejects non-object payloads before they reach the ledger.
    """

    model_config = ConfigDict(extra="allow")


class AnswerDeltaPayload(AgentEventPayload):
    text: str


class AnswerCompletedPayload(AgentEventPayload):
    answer: str


class RunEventPayload(AgentEventPayload):
    status: str | None = None
    run_id: int | None = None


class ToolEventPayload(AgentEventPayload):
    tool_name: str | None = None
    tool_call_id: str | int | None = None


class ApprovalEventPayload(AgentEventPayload):
    approval_id: str | int | None = None


class ProgressPayload(AgentEventPayload):
    text: str


_PAYLOAD_SCHEMAS: dict[str, type[AgentEventPayload]] = {
    "run_created": RunEventPayload,
    "run_paused": RunEventPayload,
    "run_resumed": RunEventPayload,
    "run_completed": RunEventPayload,
    "run_failed": RunEventPayload,
    "run_interrupted": RunEventPayload,
    "answer_started": AgentEventPayload,
    "answer_delta": AnswerDeltaPayload,
    "answer_completed": AnswerCompletedPayload,
    "visible_thought_delta": ProgressPayload,
    "visible_progress_delta": ProgressPayload,
    "tool_call_started": ToolEventPayload,
    "tool_call_completed": ToolEventPayload,
    "tool_call_failed": ToolEventPayload,
    "approval_required": ApprovalEventPayload,
    "approval_granted": ApprovalEventPayload,
    "approval_rejected": ApprovalEventPayload,
}


class AgentEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    event_seq: int
    schema_version: int = Field(default=1, ge=1)
    run_id: int
    thread_id: str
    event_type: str
    node_name: str = ""
    visibility: str
    display_channel: str
    payload: dict[str, Any]
    created_at: str


def validate_event_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    schema = _PAYLOAD_SCHEMAS.get(event_type, AgentEventPayload)
    return schema.model_validate(payload).model_dump(exclude_none=True)


def event_envelope_from_record(item: Any) -> dict[str, Any]:
    return AgentEventEnvelope(
        id=item.id,
        event_seq=item.id,
        schema_version=item.schema_version,
        run_id=item.run_id,
        thread_id=item.thread_id,
        event_type=item.event_type,
        node_name=item.node_name,
        visibility=item.visibility,
        display_channel=item.display_channel,
        payload=item.payload_json or {},
        created_at=item.created_at.isoformat() if item.created_at else "",
    ).model_dump()


def event_channels(event_type: str) -> tuple[str, str]:
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
    event_id: int | None = None,
    event_seq: int | None = None,
    created_at: str | None = None,
) -> None:
    """Push an event to the SSE stream queue. Does NOT persist to DB.

    Callers that need DB persistence should call record_event() separately.

    The *visibility* and *display_channel* fields tell the frontend which
    UI region should consume this event.  When omitted they are resolved
    automatically from the event_type.
    """
    if queue is None:
        return
    resolved_visibility, resolved_channel = event_channels(event_type)
    queue.put_nowait(
        {
            "event": event_type,
            "data": {
                "id": event_id,
                "event_seq": event_seq,
                "schema_version": 1,
                "run_id": run_id,
                "thread_id": thread_id,
                "event_type": event_type,
                "node_name": node_name,
                "visibility": visibility or resolved_visibility,
                "display_channel": display_channel or resolved_channel,
                "payload": payload,
                "created_at": created_at or datetime.now().isoformat(),
            },
        }
    )
