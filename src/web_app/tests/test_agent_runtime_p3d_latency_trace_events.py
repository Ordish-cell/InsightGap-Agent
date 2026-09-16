import inspect
import sys
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.latency import build_runtime_slow_path_hints
from src.web_app.services import agent_service
from src.web_app.tests.db_test_utils import make_test_session
from src.web_app.tests.test_chat_control import env as env


def test_slow_path_hints_cover_slow_prepare_empty_evidence_and_supervisor_fallback():
    trace = {
        "prefetch": {"elapsed_ms": 3001, "warnings": ["prefetch_timeout:rag"]},
        "parallel_read": {
            "elapsed_ms": 6001,
            "warnings": ["parallel_read_timeout"],
            "branch_timings": {"context_skill": 6001},
            "rag_prepare": {
                "elapsed_ms": 3001,
                "no_evidence": True,
            },
        },
        "supervisor": {"fallback_reason": "dispatch_mismatch"},
    }

    hints = build_runtime_slow_path_hints(trace)

    assert "prefetch_slow" in hints
    assert "prefetch_timeout_or_warning" in hints
    assert "parallel_read_slow" in hints
    assert "parallel_read_timeout_or_warning" in hints
    assert "rag_prepare_slow" in hints
    assert "context_skill_slow" in hints
    assert "rag_prepare_no_evidence" in hints
    assert "supervisor_control_fallback" in hints


def test_fast_trace_has_no_slow_path_hints():
    trace = {
        "prefetch": {"elapsed_ms": 25, "warnings": []},
        "parallel_read": {
            "elapsed_ms": 40,
            "warnings": [],
            "branch_timings": {"context_skill": 30},
            "rag_prepare": {"elapsed_ms": 15, "no_evidence": False},
        },
        "supervisor": {"fallback_reason": ""},
    }

    assert build_runtime_slow_path_hints(trace) == []


def test_latency_trace_event_payload_excludes_final_answer_text(monkeypatch):
    published = []

    monkeypatch.setattr(agent_service, "publish_event", lambda *args, **kwargs: published.append((args, kwargs)))

    state = {
        "answer": "secret answer",
        "final_answer": "secret answer",
        "final_output": "secret answer",
        "runtime_latency_trace": {"mode": "runtime_latency_trace"},
        "runtime_latency_warnings": ["parallel_read_warnings_present"],
        "runtime_slow_path_hints": ["rag_prepare_no_evidence"],
    }

    agent_service._emit_runtime_latency_trace_event(make_test_session(), None, 1, "t", 1, state)

    assert published[0][0][3] == "runtime_latency_trace"
    payload = published[0][0][4]
    assert payload == {
        "runtime_latency_trace": {"mode": "runtime_latency_trace"},
        "runtime_latency_warnings": ["parallel_read_warnings_present"],
        "runtime_slow_path_hints": ["rag_prepare_no_evidence"],
    }
    assert "secret answer" not in str(payload)


def test_agent_service_emits_latency_trace_before_terminal_run_events():
    source = inspect.getsource(agent_service.execute_prepared_run)
    trace_call = '_emit_runtime_latency_trace_event(db, stream_queue, run.id, thread_id, user_id, state)'
    trace_index = source.index(trace_call)
    assert trace_index < source.index('publish_event(db, stream_queue, run.id, "run_failed"', trace_index)
    assert trace_index < source.index('publish_event(db, stream_queue, run.id, "run_completed"', trace_index)
    assert trace_index < source.index('publish_event(db, stream_queue, run.id, "run_paused"', trace_index)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,event", [("completed", "run_completed"), ("failed", "run_failed"), ("waiting_approval", "run_paused")])
async def test_resume_path_emits_latency_before_actual_terminal_event(env, monkeypatch, status, event):
    from src.web_app.db.repositories.agent_repository import AgentEventRepository
    from src.web_app.models.orm import AgentRun
    monkeypatch.setattr("src.web_app.services.summary_tasks.summary_tasks.schedule", lambda *a: None)
    with env.factory() as db:
        state = {"user_id": env.user, "run_id": env.run, "runtime_version": 2, "conversation_id": "chat",
                 "status": status, "final_answer": "result", "thread_id": f"run:{env.run}"}
        await agent_service._finalize_resume(db=db, user_id=env.user, run_id=env.run, run=db.get(AgentRun, env.run),
            state=state, conversation_id="chat", thread_id=state["thread_id"], user_input="test",
            started_at=datetime.now(), pause_mode="interrupt", pending_approval_id=1,
            pending_tool_call_id=1, pending_tool_name="email.send", stream_queue=None)
        events = [e.event_type for e in AgentEventRepository(db).list_by_run(env.user, env.run)]
        assert events.index("runtime_latency_trace") < events.index(event)
