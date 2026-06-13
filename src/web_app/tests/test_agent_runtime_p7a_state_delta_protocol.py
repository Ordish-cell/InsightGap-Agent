import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import nodes as runtime_nodes_module
from src.web_app.agent.runtime.node_groups import eval_final_nodes, read_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.schemas import AgentResult, NodeResult, StateDelta
from src.web_app.agent.runtime.state_delta import apply_state_delta, record_node_result
from src.web_app.tests.db_test_utils import make_test_session


def test_state_delta_and_node_result_are_json_serializable():
    delta = StateDelta(
        updates={"evaluation_result": {"score": 1}},
        append={"events": [{"event": "x"}]},
        completed_node="evaluator",
        warnings=["warn"],
        events=[{"event": "trace"}],
        agent_result=AgentResult(agent="rag_agent", status="ok", confidence=0.8),
        metadata={"source": "test"},
    )
    node_result = NodeResult(node="evaluator", delta=delta, summary="done", elapsed_ms=3)

    assert "evaluation_result" in delta.model_dump_json()
    assert "evaluator" in node_result.model_dump_json()


def test_apply_state_delta_updates_appends_completion_and_agent_result():
    state = {"completed_nodes": [], "agent_results": [], "items": [1]}
    delta = StateDelta(
        updates={"foo": "bar"},
        append={"items": [2], "events": [{"event": "x"}]},
        completed_node="parallel_prefetch",
        agent_result={"agent": "rag_agent", "status": "ok", "confidence": 0.9},
        warnings=["prefetch_warn"],
    )

    result = apply_state_delta(state, delta)

    assert result["foo"] == "bar"
    assert result["items"] == [1, 2]
    assert result["events"] == [{"event": "x"}]
    assert "parallel_prefetch" in result["completed_nodes"]
    assert result["agent_results"][0]["agent"] == "rag_agent"
    assert result["node_warnings"] == ["prefetch_warn"]


def test_apply_state_delta_does_not_overwrite_protected_fields_by_default():
    state = {
        "status": "waiting_approval",
        "approval_payload": {"approval_id": "a1"},
        "pending_tool_call_id": 7,
        "final_payload": {"answer": "paused"},
        "final_output": "paused",
    }

    apply_state_delta(
        state,
        StateDelta(
            updates={
                "status": "completed",
                "approval_payload": None,
                "pending_tool_call_id": None,
                "final_payload": {},
                "final_output": "",
                "safe": True,
            }
        ),
    )

    assert state["status"] == "waiting_approval"
    assert state["approval_payload"] == {"approval_id": "a1"}
    assert state["pending_tool_call_id"] == 7
    assert state["final_payload"] == {"answer": "paused"}
    assert state["final_output"] == "paused"
    assert state["safe"] is True


def test_record_node_result_appends_structured_result():
    state = {}

    record_node_result(
        state,
        node="supervisor_observer",
        delta=StateDelta(updates={"supervisor_decision": {}}),
        summary="observed",
    )

    assert state["node_results"][0]["node"] == "supervisor_observer"
    assert state["node_results"][0]["delta"]["updates"] == {"supervisor_decision": {}}


@pytest.mark.asyncio
async def test_supervisor_observer_keeps_old_fields_and_appends_node_result():
    route_plan = {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"}
    state = {
        "route_plan": route_plan,
        "execution_plan": {"intent": "rag", "tasks": [{"agent": "rag_agent"}]},
        "completed_nodes": [],
        "status": "running",
    }

    result = await RuntimeNodes(make_test_session(), {}).supervisor_observer(state)

    assert result["supervisor_decision"]["current_route"] == ["rag_agent"]
    assert result["supervisor_warnings"] == []
    assert result["supervisor_trace"]
    assert result["node_results"][-1]["node"] == "supervisor_observer"


@pytest.mark.asyncio
async def test_parallel_prefetch_keeps_old_fields_and_appends_node_result(monkeypatch):
    async def fake_prefetch(state, db, payload):
        state["prefetch_results"] = {"rag": {"count": 1}}
        state["prefetch_warnings"] = ["prefetch_timeout:memory"]
        state["prefetch_elapsed_ms"] = 12
        state["prefetch_agent_results"] = [{"agent": "rag_prefetch"}]
        return state

    monkeypatch.setattr(read_nodes, "run_parallel_prefetch", fake_prefetch)
    monkeypatch.setattr(read_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(read_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(read_nodes, "record_step", lambda *args, **kwargs: None)

    state = {"run_id": 1, "user_id": 1, "route": "chat", "user_input": "hi"}
    result = await RuntimeNodes(make_test_session(), {}).parallel_prefetch(state)

    assert result["prefetch_results"] == {"rag": {"count": 1}}
    assert result["prefetch_warnings"] == ["prefetch_timeout:memory"]
    assert result["node_results"][-1]["node"] == "parallel_prefetch"
    assert result["node_results"][-1]["elapsed_ms"] == 12


@pytest.mark.asyncio
async def test_parallel_read_stage_keeps_old_fields_and_appends_node_result(monkeypatch):
    async def fake_parallel_read(state, nodes, payload):
        state["parallel_read_results"] = {"rag_prepare": {"status": "ok"}}
        state["parallel_read_warnings"] = ["rag_prepare_timeout"]
        state["parallel_read_elapsed_ms"] = 20
        state["parallel_read_branch_timings"] = {"rag_prepare": 20}
        return state

    monkeypatch.setattr(read_nodes, "run_parallel_read_stage", fake_parallel_read)
    monkeypatch.setattr(read_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(read_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(read_nodes, "record_step", lambda *args, **kwargs: None)

    state = {"run_id": 1, "user_id": 1, "route": "rag", "route_plan": {"route": ["rag_agent"]}}
    result = await RuntimeNodes(make_test_session(), {}).parallel_read_stage(state)

    assert result["parallel_read_results"] == {"rag_prepare": {"status": "ok"}}
    assert result["parallel_read_warnings"] == ["rag_prepare_timeout"]
    assert result["node_results"][-1]["node"] == "parallel_read_stage"
    assert result["node_results"][-1]["elapsed_ms"] == 20


@pytest.mark.asyncio
async def test_evaluator_keeps_old_fields_completion_and_appends_node_result(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)

    state = {
        "run_id": 1,
        "user_id": 1,
        "route": "rag",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": ["rag_agent"],
        "agent_results": [{"agent": "rag_agent", "status": "ok", "confidence": 0.7, "evidence": []}],
        "rag_result": {"answer": "fallback", "evidence": []},
        "final_output": "draft",
    }

    result = await RuntimeNodes(make_test_session(), {}).evaluator(state)

    assert result["evaluation_result"]["score"] == 0.65
    assert "evidence_missing" in result["final_warnings"]
    assert "evaluator" in result["completed_nodes"]
    assert result["node_results"][-1]["node"] == "evaluator"
    assert result["node_results"][-1]["delta"]["updates"]["evaluation_result"] == result["evaluation_result"]


def test_node_results_do_not_change_dispatch_decision(monkeypatch):
    from src.web_app.agent.runtime.graph import AgentRuntime

    monkeypatch.setattr(runtime_nodes_module, "record_step", lambda *args, **kwargs: None)
    runtime = AgentRuntime(make_test_session(), {})
    state = {
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "node_results": [{"node": "parallel_read_stage"}],
    }

    assert runtime._dispatch_next_route_node(state) == "rag_agent"
