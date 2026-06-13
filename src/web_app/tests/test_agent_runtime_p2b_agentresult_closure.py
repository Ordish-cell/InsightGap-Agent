import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import nodes as runtime_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.tests.db_test_utils import make_test_session


def _patch_runtime_side_effects(monkeypatch):
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_llm_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "emit_visible_thought", lambda *args, **kwargs: None)


@pytest.mark.asyncio
async def test_tool_agent_waiting_approval_agent_result_and_evaluator_constraint(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr(runtime_nodes, "llm_select_tools", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skip llm")))
    monkeypatch.setattr(runtime_nodes, "infer_tool", lambda *args, **kwargs: ("email.send", {"to": "a@example.com", "subject": "Hi", "body": "Hello"}))
    monkeypatch.setattr(runtime_nodes, "validate_tool_input", lambda tool_name, tool_input: (tool_input, []))
    monkeypatch.setattr(runtime_nodes.mcp_service, "call_tool", lambda *args, **kwargs: {
        "id": "tool-call-1",
        "status": "waiting_approval",
        "tool_name": "email.send",
        "output": {"_metadata": {"approval_id": "approval-1"}},
    })

    route_plan = {"intent": "tool.email", "route": ["tool_agent", "evaluator"], "risk_level": "L3"}
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "tool",
        "user_input": "send an email",
        "route_plan": route_plan,
        "execution_plan": execution_plan_from_route_plan(route_plan),
    }

    after_tool = await RuntimeNodes(db, {}).tool_agent(state)

    assert after_tool["status"] == "waiting_approval"
    assert after_tool["approval_required"] is True
    tool_result = after_tool["agent_results"][-1]
    assert tool_result["agent"] == "tool_agent"
    assert tool_result["status"] == "needs_approval"
    assert tool_result["tool_calls"][0]["status"] == "waiting_approval"

    evaluated = await RuntimeNodes(db, {}).evaluator(after_tool)

    assert evaluated["status"] == "waiting_approval"
    assert "tool_waiting_approval" in evaluated["final_warnings"]
    assert any("approval is required" in item for item in evaluated["final_response_constraints"])


@pytest.mark.asyncio
async def test_tool_agent_failed_and_denied_constraints(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)

    base_state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "tool",
        "route_plan": {"intent": "tool", "route": ["tool_agent", "evaluator"], "risk_level": "L3"},
    }

    failed = await RuntimeNodes(db, {}).evaluator({
        **base_state,
        "agent_results": [{"agent": "tool_agent", "status": "failed", "summary": "boom"}],
    })
    denied = await RuntimeNodes(db, {}).evaluator({
        **base_state,
        "agent_results": [{"agent": "tool_agent", "status": "denied", "summary": "blocked"}],
    })

    assert "tool_failed" in failed["final_warnings"]
    assert any("tool action has been completed" in item for item in failed["final_response_constraints"])
    assert "tool_denied" in denied["final_warnings"]
    assert any("tool action has been completed" in item for item in denied["final_response_constraints"])


@pytest.mark.asyncio
async def test_artifact_agent_failed_constraint(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr(runtime_nodes.artifact_service, "save_text_artifact", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))

    route_plan = {"intent": "artifact", "route": ["artifact_agent", "evaluator"], "risk_level": "L2"}
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "artifact",
        "user_input": "make a file",
        "route_plan": route_plan,
        "execution_plan": execution_plan_from_route_plan(route_plan),
        "final_output": "content",
    }

    after_artifact = await RuntimeNodes(db, {}).artifact_agent(state)
    artifact_result = after_artifact["agent_results"][-1]
    assert artifact_result["agent"] == "artifact_agent"
    assert artifact_result["status"] == "failed"

    evaluated = await RuntimeNodes(db, {}).evaluator(after_artifact)
    assert "artifact_missing" in evaluated["final_warnings"]
    assert any("artifact or file was generated" in item for item in evaluated["final_response_constraints"])


@pytest.mark.asyncio
async def test_artifact_agent_ok_and_final_payload_keeps_legacy_artifacts(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr(runtime_nodes.artifact_service, "save_text_artifact", lambda *args, **kwargs: "artifact.md")

    class FakeArtifactRepository:
        def __init__(self, db_arg):
            pass

        def create(self, **kwargs):
            return SimpleNamespace(id=7, artifact_type=kwargs["artifact_type"], title=kwargs["title"], file_path=kwargs["file_path"])

    monkeypatch.setattr(runtime_nodes, "ArtifactRepository", FakeArtifactRepository)

    route_plan = {"intent": "artifact", "route": ["artifact_agent", "evaluator", "final_response"], "risk_level": "L2"}
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "artifact",
        "user_input": "make a file",
        "route_plan": route_plan,
        "execution_plan": execution_plan_from_route_plan(route_plan),
        "final_output": "artifact content",
    }

    after_artifact = await RuntimeNodes(db, {}).artifact_agent(state)
    assert after_artifact["agent_results"][-1]["status"] == "ok"
    assert after_artifact["agent_results"][-1]["artifacts"][0]["id"] == 7

    final_state = await RuntimeNodes(db, {}).final_response(after_artifact)
    assert final_state["final_payload"]["artifacts"][0]["id"] == 7
    assert final_state["final_payload"]["agent_results"]


@pytest.mark.asyncio
async def test_memory_agent_failed_constraint(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)

    route_plan = {"intent": "memory", "route": ["memory_agent", "evaluator"], "risk_level": "L0"}
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "memory",
        "route_plan": route_plan,
        "agent_results": [{"agent": "memory_agent", "status": "failed", "summary": "write failed"}],
    }

    evaluated = await RuntimeNodes(db, {}).evaluator(state)

    assert "memory_write_failed" in evaluated["final_warnings"]
    assert any("memory was saved" in item for item in evaluated["final_response_constraints"])


@pytest.mark.asyncio
async def test_rag_agent_ok_without_evidence_constraint(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "route_plan": {"intent": "rag", "route": ["rag_agent", "evaluator"], "risk_level": "L1"},
        "agent_results": [{"agent": "rag_agent", "status": "ok", "summary": "answer", "evidence": []}],
    }

    evaluated = await RuntimeNodes(db, {}).evaluator(state)

    assert "rag_evidence_missing" in evaluated["final_warnings"]
    assert any("retrieved document evidence" in item for item in evaluated["final_response_constraints"])


@pytest.mark.asyncio
async def test_prefetch_results_do_not_count_as_formal_agent_success(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "route_plan": {"intent": "rag", "route": ["evaluator"], "risk_level": "L1"},
        "prefetch_results": {"rag": {"evidence": [{"id": 1}], "count": 1}},
        "prefetch_agent_results": [{"agent": "rag_prefetch", "status": "ok", "evidence": [{"id": 1}]}],
        "agent_results": [
            {"agent": "rag_prefetch", "status": "ok", "evidence": [{"id": 1}], "metadata": {"source": "prefetch"}},
            {"agent": "memory_prefetch", "status": "ok", "metadata": {"source": "prefetch"}},
            {"agent": "skill_prefetch", "status": "ok", "metadata": {"source": "prefetch"}},
        ],
    }

    evaluated = await RuntimeNodes(db, {}).evaluator(state)

    assert "rag_evidence_missing" not in evaluated["final_warnings"]
    assert not any("retrieved document evidence" in item for item in evaluated["final_response_constraints"])


def test_final_prompt_includes_runtime_constraints():
    db = make_test_session()
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "did you send it?",
        "route_plan": {"intent": "tool", "route": ["tool_agent", "final_response"], "risk_level": "L3"},
        "final_response_constraints": [
            "Do not claim the tool action has been completed. Tell the user approval is required before execution."
        ],
        "final_warnings": ["tool_waiting_approval"],
    }

    prompt = RuntimeNodes(db, {})._build_legacy_prompt(state, "", state["route_plan"])

    assert "[Runtime Safety Constraints]" in prompt
    assert "approval is required" in prompt
