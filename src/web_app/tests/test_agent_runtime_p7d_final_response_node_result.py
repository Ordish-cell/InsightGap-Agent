import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state_delta import latest_agent_result
from src.web_app.tests.db_test_utils import make_test_session


def _patch_final_side_effects(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))


def _base_state(final_output="stable final answer"):
    return {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route": "chat",
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "completed_nodes": [],
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "final_output": final_output,
        "visible_thoughts": [],
        "langgraphstatus": {},
    }


@pytest.mark.asyncio
async def test_final_response_success_appends_node_result_without_payload_node_results(monkeypatch):
    _patch_final_side_effects(monkeypatch)

    result = await RuntimeNodes(make_test_session(), {}).final_response(_base_state())

    assert result["final_answer"] == "stable final answer"
    assert result["final_output"] == "stable final answer"
    assert result["final_payload"]["answer"] == "stable final answer"
    assert "node_results" not in result["final_payload"]
    assert latest_agent_result(result, "final_response")["status"] == "ok"
    node_result = result["node_results"][-1]
    assert node_result["node"] == "final_response"
    assert node_result["status"] == "ok"
    assert node_result["delta"]["updates"]["final_answer"] == result["final_answer"]
    assert node_result["delta"]["updates"]["final_payload"] == result["final_payload"]
    assert node_result["delta"]["updates"]["runtime_latency_trace"] == result["runtime_latency_trace"]


@pytest.mark.asyncio
async def test_final_response_empty_answer_appends_failed_node_result(monkeypatch):
    _patch_final_side_effects(monkeypatch)

    async def empty_final_answer(self, state, draft_answer):
        return ""

    monkeypatch.setattr(eval_final_nodes.EvalFinalNodesMixin, "_generate_final_answer_with_llm", empty_final_answer)

    result = await RuntimeNodes(make_test_session(), {}).final_response(_base_state(final_output=""))

    assert result["final_answer"] == ""
    assert result["final_payload"]["answer"] == ""
    assert latest_agent_result(result, "final_response")["status"] == "failed"
    assert latest_agent_result(result, "final_response")["errors"] == ["final_answer_empty"]
    assert result["node_results"][-1]["node"] == "final_response"
    assert result["node_results"][-1]["status"] == "failed"
    assert "node_results" not in result["final_payload"]


@pytest.mark.asyncio
async def test_existing_node_results_do_not_influence_final_answer(monkeypatch):
    _patch_final_side_effects(monkeypatch)
    state = _base_state(final_output="clean user-facing answer")
    state["node_results"] = [{
        "node": "previous_node",
        "status": "ok",
        "delta": {"updates": {"final_answer": "poisoned answer"}},
    }]

    result = await RuntimeNodes(make_test_session(), {}).final_response(state)

    assert result["final_answer"] == "clean user-facing answer"
    assert result["final_payload"]["answer"] == "clean user-facing answer"
    assert result["node_results"][0]["node"] == "previous_node"
    assert result["node_results"][-1]["node"] == "final_response"
    assert "node_results" not in result["final_payload"]


@pytest.mark.asyncio
async def test_streaming_guard_flags_are_preserved_when_llm_path_used(monkeypatch):
    _patch_final_side_effects(monkeypatch)

    async def llm_final_answer(self, state, draft_answer):
        state["_answer_delta_emitted"] = True
        state["_answer_completed_emitted"] = True
        return "llm answer"

    monkeypatch.setattr(eval_final_nodes.EvalFinalNodesMixin, "_generate_final_answer_with_llm", llm_final_answer)

    result = await RuntimeNodes(make_test_session(), {}).final_response(_base_state(final_output=""))

    assert result["final_answer"] == "llm answer"
    assert result["_answer_delta_emitted"] is True
    assert result["_answer_completed_emitted"] is True
    assert result["node_results"][-1]["node"] == "final_response"
