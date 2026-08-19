import asyncio
import json

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from src.web_app.agent.runtime import event_ledger
from src.web_app.agent.runtime.emitter import RuntimeEventEmitter
from src.web_app.agent.runtime.event_ledger import publish_event
from src.web_app.agent.runtime.ledger_stream import stream_ledger_events
from src.web_app.db.repositories.agent_repository import AgentEventRepository
from src.web_app.models.orm import AgentRun, User
from src.web_app.services.agent_service import _stream_answer_deltas, replay_events
from src.web_app.tests.db_test_utils import make_test_session


def _run(db):
    user = User(email="event-ledger@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    run = AgentRun(user_id=user.id, thread_id="thread-ledger", user_input="test")
    db.add(run)
    db.commit()
    db.refresh(run)
    return user, run


def test_publish_event_persists_before_projecting_to_sse():
    db = make_test_session()
    user, run = _run(db)
    queue = asyncio.Queue()

    event = publish_event(
        db,
        queue,
        run.id,
        "run_completed",
        {"status": "completed"},
        user_id=user.id,
        thread_id=run.thread_id,
    )

    persisted = AgentEventRepository(db).list_by_run(user.id, run.id)
    streamed = queue.get_nowait()
    assert event is not None
    assert [item.id for item in persisted] == [event.id]
    assert streamed["data"]["id"] == event.id
    assert streamed["data"]["event_seq"] == event.id
    assert streamed["data"]["payload"] == event.payload_json
    assert streamed["data"]["created_at"] == event.created_at.isoformat()


def test_publish_event_does_not_project_when_persistence_fails(monkeypatch):
    db = make_test_session()
    user, run = _run(db)
    queue = asyncio.Queue()

    def fail_persistence(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(event_ledger, "record_event", fail_persistence)
    with pytest.raises(RuntimeError, match="database unavailable"):
        publish_event(db, queue, run.id, "run_failed", {}, user_id=user.id)

    assert queue.empty()


@pytest.mark.asyncio
async def test_runtime_emitter_projects_the_persisted_event_identity():
    db = make_test_session()
    user, run = _run(db)
    queue = asyncio.Queue()
    emitter = RuntimeEventEmitter(
        db,
        {"run_id": run.id, "user_id": user.id, "thread_id": run.thread_id},
        queue,
    )

    await emitter.status("run_paused", {"status": "waiting_approval"})

    event = AgentEventRepository(db).list_by_run(user.id, run.id)[0]
    streamed = queue.get_nowait()
    assert streamed["data"]["event_seq"] == event.id
    assert streamed["data"]["payload"] == event.payload_json


@pytest.mark.asyncio
async def test_answer_stream_is_a_projection_of_the_persisted_ledger():
    db = make_test_session()
    user, run = _run(db)
    queue = asyncio.Queue()

    await _stream_answer_deltas(
        db,
        queue,
        run.id,
        run.thread_id,
        user.id,
        "A short answer.",
    )

    persisted = AgentEventRepository(db).list_by_run(user.id, run.id)
    streamed = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [item.event_type for item in persisted] == ["answer_started", "answer_delta", "answer_completed"]
    assert [item["data"]["event_seq"] for item in streamed] == [item.id for item in persisted]
    assert [item["data"]["payload"] for item in streamed] == [item.payload_json for item in persisted]


def test_replay_events_uses_a_stable_cursor_page():
    db = make_test_session()
    user, run = _run(db)
    for index in range(5):
        publish_event(
            db,
            None,
            run.id,
            "visible_progress_delta",
            {"text": f"step {index}"},
            user_id=user.id,
            thread_id=run.thread_id,
        )

    first = replay_events(db, user.id, run.id, limit=2)
    publish_event(db, None, run.id, "visible_progress_delta", {"text": "late"}, user_id=user.id)
    second = replay_events(
        db,
        user.id,
        run.id,
        after_seq=first["next_seq"],
        until_seq=first["until_seq"],
        limit=10,
    )

    assert [event["payload"]["text"] for event in first["events"]] == ["step 0", "step 1"]
    assert first["has_more"] is True
    assert [event["payload"]["text"] for event in second["events"]] == ["step 2", "step 3", "step 4"]
    assert second["until_seq"] == first["until_seq"]
    assert all(event["payload"]["text"] != "late" for event in second["events"])
    assert second["next_seq"] > first["next_seq"]
    assert all(event["event_seq"] == event["id"] for event in first["events"] + second["events"])
    assert all(event["display_channel"] == "thinking" for event in first["events"])


def test_replay_events_filters_by_event_type_without_step_fallback():
    db = make_test_session()
    user, run = _run(db)
    publish_event(db, None, run.id, "run_created", {}, user_id=user.id)
    publish_event(db, None, run.id, "run_completed", {}, user_id=user.id)

    replay = replay_events(db, user.id, run.id, event_type="run_completed")

    assert [event["event_type"] for event in replay["events"]] == ["run_completed"]
    assert replay["has_more"] is False


def test_invalid_core_payload_fails_before_persistence_or_projection():
    db = make_test_session()
    user, run = _run(db)
    queue = asyncio.Queue()

    with pytest.raises(ValidationError):
        publish_event(db, queue, run.id, "answer_delta", {}, user_id=user.id)

    assert AgentEventRepository(db).list_by_run(user.id, run.id) == []
    assert queue.empty()


@pytest.mark.asyncio
async def test_ledger_stream_backfills_with_matching_sse_ids_and_closes_on_terminal():
    db = make_test_session()
    user, run = _run(db)
    first = publish_event(db, None, run.id, "visible_progress_delta", {"text": "one"}, user_id=user.id)
    second = publish_event(db, None, run.id, "answer_delta", {"text": "two"}, user_id=user.id)
    terminal = publish_event(db, None, run.id, "run_completed", {"status": "completed"}, user_id=user.id)
    factory = sessionmaker(bind=db.bind, future=True)

    chunks = [chunk async for chunk in stream_ledger_events(factory, user.id, run.id, after_seq=first.id)]
    envelopes = [json.loads(next(line[6:] for line in chunk.splitlines() if line.startswith("data: "))) for chunk in chunks]

    assert [item["event_seq"] for item in envelopes] == [second.id, terminal.id]
    assert [int(next(line[4:] for line in chunk.splitlines() if line.startswith("id: "))) for chunk in chunks] == [second.id, terminal.id]


@pytest.mark.asyncio
async def test_ledger_heartbeat_is_not_persisted():
    db = make_test_session()
    user, run = _run(db)
    factory = sessionmaker(bind=db.bind, future=True)
    stream = stream_ledger_events(factory, user.id, run.id, poll_interval=0, heartbeat_interval=0)

    assert await anext(stream) == ": heartbeat\n\n"
    await stream.aclose()
    assert AgentEventRepository(db).list_by_run(user.id, run.id) == []
