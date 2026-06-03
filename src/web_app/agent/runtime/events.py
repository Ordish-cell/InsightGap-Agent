import json
from typing import Any, Iterable


def to_sse(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        event_name = event.get("event", "status")
        data = json.dumps(event.get("data", event), ensure_ascii=False)
        yield f"event: {event_name}\ndata: {data}\n\n"
