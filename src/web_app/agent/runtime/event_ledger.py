"""Persist Agent events before projecting them to live consumers."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.agent.runtime.events import queue_stream_event
from src.web_app.models.orm import AgentEvent


def publish_event(
    db: Session,
    queue: Any,
    run_id: int,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    node_name: str = "",
    user_id: int | None = None,
    thread_id: str | None = None,
    visibility: str | None = None,
    display_channel: str | None = None,
) -> AgentEvent | None:
    """Append once to the ledger, then project that persisted event to SSE."""
    event = record_event(
        db,
        run_id,
        event_type,
        payload,
        node_name=node_name,
        user_id=user_id,
        thread_id=thread_id,
        visibility=visibility,
        display_channel=display_channel,
    )
    if event is None:
        return None

    queue_stream_event(
        queue,
        event.event_type,
        event.payload_json or {},
        run_id=event.run_id,
        thread_id=event.thread_id,
        node_name=event.node_name,
        visibility=event.visibility,
        display_channel=event.display_channel,
        event_id=event.id,
        event_seq=event.id,
        created_at=event.created_at.isoformat() if event.created_at else None,
    )
    return event
