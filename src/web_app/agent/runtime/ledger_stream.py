"""SSE tail backed exclusively by the persisted Agent event ledger."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from time import monotonic

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.events import event_envelope_from_record
from src.web_app.db.repositories.agent_repository import AgentEventRepository


TERMINAL_EVENT_TYPES = {"run_completed", "run_failed", "run_interrupted", "run_paused"}


async def stream_ledger_events(
    session_factory: Callable[[], Session],
    user_id: int,
    run_id: int,
    *,
    after_seq: int = 0,
    poll_interval: float = 0.25,
    heartbeat_interval: float = 15.0,
) -> AsyncIterator[str]:
    """Catch up from PostgreSQL, then tail until a terminal event is drained."""
    cursor = after_seq
    last_output = monotonic()
    terminal_seen = False

    while True:
        with session_factory() as db:
            rows = AgentEventRepository(db).list_replay(
                user_id,
                run_id,
                after_seq=cursor,
                limit=200,
            )

        for row in rows:
            envelope = event_envelope_from_record(row)
            cursor = row.id
            terminal_seen = terminal_seen or row.event_type in TERMINAL_EVENT_TYPES
            data = json.dumps(envelope, ensure_ascii=False, default=str)
            yield f"id: {row.id}\nevent: {row.event_type}\ndata: {data}\n\n"
            last_output = monotonic()

        if terminal_seen and len(rows) < 200:
            return

        if not rows and monotonic() - last_output >= heartbeat_interval:
            yield ": heartbeat\n\n"
            last_output = monotonic()

        await asyncio.sleep(poll_interval)
