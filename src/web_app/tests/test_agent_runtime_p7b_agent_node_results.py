import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.node_groups import agent_nodes, eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.state_delta import latest_agent_result, record_agent_node_result
from src.web_app.tests.db_test_utils import make_test_session


def _patch_common(monkeypatch):
    monkeypatch.setattr(agent_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))


def _base_state(route="rag", route_plan=None):
    return {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": route,
        "user_input": "question",
        "route_plan": route_plan or {"intent": route, "route": [f"{route}_agent"], "risk_level": "L1"},
        "completed_nodes": [],
        "agent_results": [],
    }


def test_latest_agent_result_and_record_agent_node_result_helper():
    state = {
        "agent_results": [
            {"agent": "rag_agent", "status": "failed", "summary": "old"},
            {"agent": "rag_agent", "status": "ok", "summary": "new"},
        ]
    }

    assert latest_agent_result(state, "rag_agent")["summary"] == "new"
    record_agent_node_result(state, node="rag_agent", updates={"rag_result": {"answer": "a"}})

    node_result = state["node_results"][-1]
    assert node_result["node"] == "rag_agent"
    assert node_result["status"] == "ok"
    assert node_result["delta"]["agent_result"]["summary"] == "new"
    assert node_result["delta"]["updates"] == {"rag_result": {"answer": "a"}}


@pytest.mark.asyncio
async def test_rag_agent_success_appends_node_result(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes.rag_service, "ask", lambda *args, **kwargs: {
        "answer": "rag answer",
        "answer_mode": "extractive",
        "evidence": [{"id": "e1"}],
    })

    state = _base_state(route="rag", route_plan={"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"})
    result = await RuntimeNodes(make_test_session(), {}).rag_agent(state)

    assert result["rag_result"]["answer"] == "rag answer"
    assert latest_agent_result(result, "rag_agent")["status"] == "ok"
    assert "rag_agent" in result["completed_nodes"]
    node_result = result["node_results"][-1]
    assert node_result["node"] == "rag_agent"
    assert node_result["delta"]["updates"]["rag_result"] == result["rag_result"]


@pytest.mark.asyncio
async def test_rag_agent_failure_appends_failed_node_result(monkeypatch):
    _patch_common(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("rag exploded")

    monkeypatch.setattr(agent_nodes.rag_service, "ask", boom)

    state = _base_state(route="rag", route_plan={"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"})
    result = await RuntimeNodes(make_test_session(), {}).rag_agent(state)

    assert result["rag_result"]["error"] == "rag exploded"
    node_result = result["node_results"][-1]
    assert node_result["node"] == "rag_agent"
    assert node_result["status"] == "failed"


@pytest.mark.asyncio
async def test_artifact_agent_success_and_failure_append_node_result(monkeypatch):
    _patch_common(monkeypatch)

    class FakeArtifactRepository:
        def __init__(self, db):
            pass

        def create(self, **kwargs):
            return SimpleNamespace(id=9, artifact_type=kwargs["artifact_type"], title="Artifact title", file_path="/tmp/a.md")

    monkeypatch.setattr(agent_nodes, "ArtifactRepository", FakeArtifactRepository)
    monkeypatch.setattr(agent_nodes.artifact_service, "save_text_artifact", lambda *args, **kwargs: "/tmp/a.md")

    state = _base_state(route="artifact", route_plan={"intent": "artifact", "route": ["artifact_agent"], "risk_level": "L1"})
    result = await RuntimeNodes(make_test_session(), {}).artifact_agent(state)

    assert result["artifact_result"]["id"] == 9
    assert result["node_results"][-1]["node"] == "artifact_agent"
    assert result["node_results"][-1]["status"] == "ok"

    def fail_save(*args, **kwargs):
        raise RuntimeError("artifact exploded")

    monkeypatch.setattr(agent_nodes.artifact_service, "save_text_artifact", fail_save)
    failed_state = _base_state(route="artifact", route_plan={"intent": "artifact", "route": ["artifact_agent"], "risk_level": "L1"})
    failed = await RuntimeNodes(make_test_session(), {}).artifact_agent(failed_state)

    assert failed["artifact_result"]["error"] == "artifact exploded"
    assert failed["node_results"][-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_memory_agent_explicit_success_and_failure_append_node_result(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes.AgentNodesMixin, "_is_explicit_memory_write", lambda self, text: True)
    monkeypatch.setattr(agent_nodes.AgentNodesMixin, "_extract_memory_from_user_input", lambda self, text: "memory fact")

    monkeypatch.setattr(agent_nodes.memory_service, "add_memory", lambda **kwargs: {
        "ok": True,
        "id": 3,
        "qdrant_point_id": "q1",
        "qdrant_indexed": True,
        "category": "preference",
    })
    state = _base_state(route="memory", route_plan={"intent": "memory", "route": ["memory_agent"], "risk_level": "L1"})
    result = await RuntimeNodes(make_test_session(), {}).memory_agent(state)

    assert result["memory_result"]["saved_count"] == 1
    assert result["node_results"][-1]["node"] == "memory_agent"
    assert result["node_results"][-1]["status"] == "ok"

    monkeypatch.setattr(agent_nodes.memory_service, "add_memory", lambda **kwargs: {
        "ok": False,
        "error": "memory failed",
    })
    failed_state = _base_state(route="memory", route_plan={"intent": "memory", "route": ["memory_agent"], "risk_level": "L1"})
    failed = await RuntimeNodes(make_test_session(), {}).memory_agent(failed_state)

    assert failed["memory_write_result"]["success"] is False
    assert failed["node_results"][-1]["node"] == "memory_agent"
    assert failed["node_results"][-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_skill_agent_success_appends_node_result(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_nodes.skill_service, "evaluate_reusability", lambda state: {
        "reusable_score": 0.6,
        "should_create": False,
        "reason": "reusable enough",
    })

    state = _base_state(route="skill", route_plan={"intent": "skill", "route": ["skill_agent"], "risk_level": "L1"})
    result = await RuntimeNodes(make_test_session(), {}).skill_agent(state)

    assert result["skill_result"]["reusable_score"] == 0.6
    assert result["node_results"][-1]["node"] == "skill_agent"
    assert result["node_results"][-1]["delta"]["updates"]["skill_result"] == result["skill_result"]


@pytest.mark.asyncio
async def test_research_agent_fallback_success_and_failure_append_node_result(monkeypatch):
    _patch_common(monkeypatch)

    async def fake_research_query(db, user_id, request):
        assert request.force_engine == "fallback"
        return {"status": "completed", "summary": "research summary", "findings": ["f1"], "evidence": []}

    monkeypatch.setattr(agent_nodes.research_service, "research_query", fake_research_query)

    state = _base_state(route="research", route_plan={
        "intent": "research",
        "route": ["research_agent"],
        "risk_level": "L1",
        "explicit_research": False,
    })
    result = await RuntimeNodes(make_test_session(), {}).research_agent(state)

    assert result["research_result"]["summary"] == "research summary"
    assert result["node_results"][-1]["node"] == "research_agent"
    assert result["node_results"][-1]["status"] == "ok"

    async def failed_research_query(db, user_id, request):
        raise RuntimeError("research failed")

    monkeypatch.setattr(agent_nodes.research_service, "research_query", failed_research_query)
    failed_state = _base_state(route="research", route_plan={
        "intent": "research",
        "route": ["research_agent"],
        "risk_level": "L1",
        "explicit_research": False,
    })
    failed = await RuntimeNodes(make_test_session(), {}).research_agent(failed_state)

    assert failed["node_results"][-1]["node"] == "research_agent"
    assert failed["node_results"][-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_skipped_agent_records_node_result_without_changing_completion_semantics():
    state = _base_state(route="approval", route_plan={"intent": "rag", "route": ["rag_agent"], "risk_level": "L3"})
    result = await RuntimeNodes(make_test_session(), {}).rag_agent(state)

    assert "rag_agent" in result["completed_nodes"]
    assert result["node_results"][-1]["node"] == "rag_agent"
    assert result["node_results"][-1]["status"] == "skipped"


@pytest.mark.asyncio
async def test_tool_agent_records_skipped_node_result_after_p7c(monkeypatch):
    _patch_common(monkeypatch)
    state = _base_state(route="approval", route_plan={"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"})

    result = await RuntimeNodes(make_test_session(), {}).tool_agent(state)

    assert "tool_agent" in result["completed_nodes"]
    assert result["node_results"][-1]["node"] == "tool_agent"
    assert result["node_results"][-1]["status"] == "skipped"


@pytest.mark.asyncio
async def test_final_response_payload_does_not_include_node_results(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route": "chat",
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "final_output": "answer",
        "node_results": [{"node": "previous_node"}],
    }

    result = await RuntimeNodes(make_test_session(), {}).final_response(state)

    assert result["final_payload"]["answer"] == "answer"
    assert "node_results" not in result["final_payload"]
    assert result["node_results"][-1]["node"] == "final_response"
