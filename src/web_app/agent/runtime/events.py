import json
from datetime import datetime
from typing import Any, Iterable


def to_sse(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        event_name = event.get("event", "status")
        data = json.dumps(event.get("data", event), ensure_ascii=False)
        yield f"event: {event_name}\ndata: {data}\n\n"


def queue_stream_event(
    queue: Any,
    event_type: str,
    payload: dict[str, Any],
    *,
    run_id: int | None = None,
    thread_id: str = "",
    node_name: str = "",
) -> None:
    """Push an event to the SSE stream queue. Does NOT persist to DB.

    Callers that need DB persistence should call record_event() separately.
    """
    if queue is None:
        return
    queue.put_nowait(
        {
            "event": event_type,
            "data": {
                "run_id": run_id,
                "thread_id": thread_id,
                "event_type": event_type,
                "node_name": node_name,
                "payload": payload,
                "created_at": datetime.now().isoformat(),
            },
        }
    )
