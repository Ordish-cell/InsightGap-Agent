import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.latency import build_runtime_slow_path_hints
from src.web_app.services import agent_service
from src.web_app.tests.db_test_utils import make_test_session


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


def test_resume_path_emits_latency_trace_before_run_completed():
    module_source = inspect.getsource(agent_service)
    source = module_source[
        module_source.index("async def resume_run_after_approval"):
        module_source.index("def get_run")
    ]
    trace_call = '_emit_runtime_latency_trace_event(db, stream_queue, run_id, thread_id, user_id, state)'
    assert source.index(trace_call) < source.index('publish_event(db, stream_queue, run_id, "run_completed"')
