import asyncio
import json
import os
from time import perf_counter

import pytest
from sqlalchemy import select

from src.web_app.agent.llm.context import ModelExecutionContext
from src.web_app.agent.runtime.event_ledger import publish_event
from src.web_app.agent.runtime.ledger_stream import stream_ledger_events
from src.web_app.models.orm import AgentChatMessage, AgentEvent, AgentRun
from src.web_app.services.agent_service import execute_prepared_run
from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_chat_fast_path import fake_model


@pytest.mark.asyncio
async def test_service_commits_answer_and_terminal_before_summary(env, monkeypatch):
    fake_model(monkeypatch, ["现在先休息一下。"])
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    ctx = ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "fake", "Fake")
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_run_model_context", lambda *a, **k: ctx)
    scheduled = []
    def schedule(*args):
        with env.factory() as check:
            assert check.get(AgentRun, env.run).status == "completed"
            assert check.scalar(select(AgentChatMessage).where(AgentChatMessage.role == "assistant")).content == "现在先休息一下。"
            events = check.execute(select(AgentEvent).order_by(AgentEvent.id)).scalars().all()
            assert events[-1].event_type == "run_completed"
            assert len([e for e in events if e.event_type == "answer_completed"]) == 1
        scheduled.append(args)
    monkeypatch.setattr("src.web_app.services.summary_tasks.summary_tasks.schedule", schedule)
    with env.factory() as db:
        result = await execute_prepared_run(db, env.user, env.run, {})
        assert result["status"] == "completed"
    assert scheduled == [(env.user, "chat")]


@pytest.mark.asyncio
async def test_notify_wakes_long_poll_and_replay_is_identical(env):
    stream = stream_ledger_events(env.factory, env.user, env.run, poll_interval=10)
    waiting = asyncio.create_task(anext(stream))
    await asyncio.sleep(.01)
    started = perf_counter()
    with env.factory() as db:
        event = publish_event(db, None, env.run, "progress_completed", {"text": "有依据的发现", "block_id": "finding"}, user_id=env.user)
        seq = event.id
    frame = await asyncio.wait_for(waiting, .3)
    assert (perf_counter() - started) < .3
    await stream.aclose()
    replay = stream_ledger_events(env.factory, env.user, env.run)
    assert await anext(replay) == frame
    await replay.aclose()
    with env.factory() as db:
        publish_event(db, None, env.run, "run_completed", {"status": "completed"}, user_id=env.user)
    frames = [f async for f in stream_ledger_events(env.factory, env.user, env.run, after_seq=seq)]
    assert len(frames) == 1 and 'run_completed' in frames[0]


@pytest.mark.asyncio
async def test_fake_model_latency_samples(env, monkeypatch):
    from src.web_app.tests.test_chat_fast_path import run_native
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    fake_model(monkeypatch, ["简短回答。"])
    samples = []
    with env.factory() as db:
        for _ in range(30):
            db.query(AgentEvent).delete()
            db.commit()
            started = perf_counter()
            await run_native(SupervisorNodes(db, {}), {"run_id": env.run, "user_id": env.user,
                "conversation_id": "chat", "user_input": "唉，我好累"})
            samples.append((perf_counter() - started) * 1000)
    p95 = sorted(samples)[28]
    print("CHAT_FAKE_LATENCY " + json.dumps({"samples": 30, "entry_total_p95_ms": round(p95, 1)}))
    assert p95 <= 500


@pytest.mark.asyncio
async def test_completed_turn_latency_samples(env, monkeypatch):
    from src.web_app.db.repositories.agent_repository import AgentEventRepository
    observed = {}
    original_create = AgentEventRepository.create
    def timed_create(self, **kwargs):
        row = original_create(self, **kwargs)
        observed[(row.run_id, row.event_type, (row.payload_json or {}).get("stage"))] = perf_counter()
        return row
    monkeypatch.setattr(AgentEventRepository, "create", timed_create)
    fake_model(monkeypatch, ["简短回答。"])
    monkeypatch.setattr("src.web_app.agent.runtime.graph.settings.agent_langgraph_checkpointer_enabled", False)
    ctx = ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "fake", "Fake")
    monkeypatch.setattr("src.web_app.services.llm_registry_service.resolve_run_model_context", lambda *a, **k: ctx)
    monkeypatch.setattr("src.web_app.services.summary_tasks.summary_tasks.schedule", lambda *args: None)
    prepare, complete = [], []
    with env.factory() as db:
        for index in range(30):
            run = AgentRun(user_id=env.user, conversation_id="chat", thread_id=f"sample-{index}",
                           user_input="你好", status="created", graph_state={})
            db.add(run); db.flush()
            for role in ("user", "assistant"):
                db.add(AgentChatMessage(user_id=env.user, run_id=run.id, conversation_id="chat", thread_id=run.thread_id,
                    message_id=f"sample-{index}-{role}", role=role, content="你好" if role == "user" else "",
                    status="completed" if role == "user" else "thinking"))
            db.commit()
            await execute_prepared_run(db, env.user, run.id, {})
            events = db.execute(select(AgentEvent).where(AgentEvent.run_id == run.id).order_by(AgentEvent.id)).scalars().all()
            begin = next(e for e in events if e.event_type == "run_started")
            model = next(e for e in events if e.event_type == "chat_latency" and e.payload_json["stage"] == "model_started")
            saved = next(e for e in events if e.event_type == "answer_completed")
            end = next(e for e in events if e.event_type == "run_completed")
            # SQLite's server timestamp has second precision: use monotonic commit observations.
            clock = lambda event: observed[(event.run_id, event.event_type, (event.payload_json or {}).get("stage"))]
            prepare.append((clock(model) - clock(begin)) * 1000)
            complete.append((clock(end) - clock(saved)) * 1000)
    result = {"samples": 30, "before_model_p95_ms": round(sorted(prepare)[28], 1), "saved_to_terminal_p95_ms": round(sorted(complete)[28], 1)}
    print("CHAT_TURN_LATENCY " + json.dumps(result))
    assert result["before_model_p95_ms"] <= 500
    assert result["saved_to_terminal_p95_ms"] <= 300


@pytest.mark.asyncio
@pytest.mark.skipif(os.environ.get("CHAT_REAL_SMOKE") != "1", reason="Explicit opt-in: read configured model; write isolated test DB only")
async def test_real_provider_isolated_smoke(env, monkeypatch):
    from src.web_app.db.session import SessionLocal
    from src.web_app.services.llm_registry_service import resolve_run_model_context
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.tests.test_chat_fast_path import run_native
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    with SessionLocal() as source:
        run = source.scalar(select(AgentRun).where(AgentRun.status == "completed").order_by(AgentRun.id.desc()))
        if not run:
            pytest.skip("No configured completed run")
        ctx = resolve_run_model_context(source, run.user_id, (run.graph_state or {}).get("model_context") or {})
    results = []
    previous_seq = 0
    with use_model_context(ctx), env.factory() as db:
        for question, route in [("唉，我好累", "chat"), ("请联网核查今天的科技新闻，引用来源", "workflow")]:
            db.query(AgentEvent).delete()
            db.commit()
            started = perf_counter()
            result = await run_native(SupervisorNodes(db, {}), {"run_id": env.run, "user_id": env.user,
                "conversation_id": "isolated-smoke", "user_input": question})
            events = db.execute(select(AgentEvent).where(AgentEvent.run_id == env.run, AgentEvent.id > previous_seq).order_by(AgentEvent.id)).scalars().all()
            previous_seq = events[-1].id
            assert not any(e.event_type == "chat_route_fallback" for e in events), result.get("error")
            assert result["status"] == "completed"
            if route == "workflow":
                assert any(r["capability"] == "tool" for r in result["observations"])
            results.append({"route": route, "elapsed_ms": round((perf_counter() - started) * 1000),
                            "answer_chars": len(result.get("final_output", "")),
                            "stages_ms": {e.payload_json["stage"]: round(e.payload_json["elapsed_ms"]) for e in events if e.event_type == "chat_latency"}})
    print("CHAT_REAL_SMOKE " + json.dumps({"model": ctx.model, "results": results}))
