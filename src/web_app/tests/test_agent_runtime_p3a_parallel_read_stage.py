import asyncio
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import nodes as runtime_nodes
from src.web_app.agent.runtime import parallel_read as parallel_read_module
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.parallel_read import parallel_read_stage
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.tests.db_test_utils import make_test_session


def _patch_runtime_side_effects(monkeypatch):
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "emit_visible_thought", lambda *args, **kwargs: None)


@pytest.mark.asyncio
async def test_parallel_read_stage_runs_context_and_rag_prepare_concurrently(monkeypatch):
    db = make_test_session()

    async def fake_context(state, nodes, payload):
        await asyncio.sleep(0.3)
        branch = dict(state)
        branch["context"] = {"gssc_context": "ctx"}
        return branch

    async def fake_rag_prepare(state, payload):
        await asyncio.sleep(0.3)
        return {"status": "ok", "evidence": [{"id": 1}], "elapsed_ms": 300}

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", fake_context)
    monkeypatch.setattr(parallel_read_module, "_rag_prepare_branch", fake_rag_prepare)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "user_input": "knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
    }

    started = time.perf_counter()
    result = await parallel_read_stage(state, RuntimeNodes(db, {}))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.55
    assert result["context"]["gssc_context"] == "ctx"
    assert result["parallel_read_results"]["rag_prepare"]["evidence"][0]["id"] == 1


@pytest.mark.asyncio
async def test_parallel_read_stage_runs_context_for_normal_routes(monkeypatch):
    db = make_test_session()
    calls = []

    async def fake_context(state, nodes, payload):
        calls.append((state.get("route_plan") or {}).get("intent"))
        branch = dict(state)
        branch["context"] = {"gssc_context": f"ctx:{calls[-1]}"}
        return branch

    async def forbidden_rag_prepare(state, payload):
        raise AssertionError("rag prepare should not run")

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", fake_context)
    monkeypatch.setattr(parallel_read_module, "_rag_prepare_branch", forbidden_rag_prepare)

    for intent in ("chat", "tool.email", "artifact", "memory", "research"):
        state = {
            "user_id": 1,
            "run_id": 1,
            "thread_id": "t",
            "route": "tool" if intent.startswith("tool") else intent,
            "user_input": "hello",
            "route_plan": {"intent": intent, "route": ["tool_agent"] if intent.startswith("tool") else [], "risk_level": "L1"},
        }
        result = await parallel_read_stage(state, RuntimeNodes(db, {}))
        assert result["context"]["gssc_context"] == f"ctx:{intent}"
        assert "rag_prepare" not in result.get("parallel_read_results", {})

    assert calls == ["chat", "tool.email", "artifact", "memory", "research"]


@pytest.mark.asyncio
async def test_context_skill_failure_keeps_context_fallback(monkeypatch):
    db = make_test_session()

    async def failed_context(state, nodes, payload):
        raise RuntimeError("context down")

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", failed_context)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "chat",
        "user_input": "hello",
        "route_plan": {"intent": "chat", "route": [], "risk_level": "L0"},
    }

    result = await parallel_read_stage(state, RuntimeNodes(db, {}))

    assert result["context"] == {}
    assert any(warning.startswith("context_skill_failed") for warning in result["parallel_read_warnings"])


@pytest.mark.asyncio
async def test_rag_route_prepares_evidence_without_completing_rag_agent(monkeypatch):
    db = make_test_session()

    async def fake_context(state, nodes, payload):
        branch = dict(state)
        branch["context"] = {"gssc_context": "ctx"}
        return branch

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", fake_context)
    monkeypatch.setattr(parallel_read_module.rag_service, "search_evidence", lambda *args, **kwargs: [{"id": "prepared", "quote": "prepared evidence"}])

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "user_input": "knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
    }

    result = await parallel_read_stage(state, RuntimeNodes(db, {}))

    assert result["parallel_read_results"]["rag_prepare"]["status"] == "ok"
    assert result["parallel_read_results"]["rag_prepare"]["evidence"][0]["id"] == "prepared"
    assert "rag_agent" not in result.get("completed_nodes", [])
    assert not any(item.get("agent") == "rag_agent" for item in result.get("agent_results", []))
    assert "rag_result" not in result


@pytest.mark.asyncio
async def test_rag_agent_reuses_prepared_evidence_without_recalling_search(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr(runtime_nodes.rag_service, "ask", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rag ask recalled")))

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "rag", "route": ["rag_agent"]}),
        "parallel_read_results": {
            "rag_prepare": {
                "status": "ok",
                "evidence": [{"document_id": 1, "chunk_id": "c1", "quote": "prepared quote", "source_title": "Doc"}],
            }
        },
    }

    result = await RuntimeNodes(db, {}).rag_agent(state)

    assert result["rag_result"]["_parallel_read_evidence_used"] is True
    assert result["rag_result"]["evidence"][0]["chunk_id"] == "c1"
    assert result["agent_results"][-1]["agent"] == "rag_agent"
    assert result["agent_results"][-1]["status"] == "ok"
    assert "rag_agent" in result["completed_nodes"]


@pytest.mark.asyncio
async def test_rag_prepare_timeout_then_formal_rag_agent_falls_back(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)

    async def slow_rag_prepare(state, payload):
        await asyncio.sleep(0.05)
        return {"status": "ok", "evidence": [{"id": "too-late"}]}

    async def fake_context(state, nodes, payload):
        return dict(state, context={"gssc_context": "ctx"})

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", fake_context)
    monkeypatch.setattr(parallel_read_module, "_rag_prepare_branch", slow_rag_prepare)
    monkeypatch.setattr(runtime_nodes.rag_service, "ask", lambda *args, **kwargs: {
        "answer": "fallback answer",
        "evidence": [{"id": "formal"}],
    })

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "user_input": "knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "rag", "route": ["rag_agent"]}),
    }

    prepared = await parallel_read_stage(state, RuntimeNodes(db, {}), rag_prepare_timeout=0.01)

    assert "rag_prepare_timeout" in prepared["parallel_read_warnings"]
    assert "rag_agent" not in prepared.get("completed_nodes", [])

    result = await RuntimeNodes(db, {}).rag_agent(prepared)
    assert result["rag_result"]["answer"] == "fallback answer"
    assert result["rag_result"]["evidence"][0]["id"] == "formal"


@pytest.mark.asyncio
async def test_parallel_read_stage_does_not_call_forbidden_agents(monkeypatch):
    db = make_test_session()
    nodes = RuntimeNodes(db, {})

    for name in ("tool_agent", "artifact_agent", "memory_agent", "skill_agent", "research_agent"):
        monkeypatch.setattr(nodes, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(f"{name} called")))

    async def fake_context(state, nodes_arg, payload):
        return dict(state, context={"gssc_context": "ctx"})

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", fake_context)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "user_input": "knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "prefetch_results": {"rag": {"evidence": [{"id": 1}]}},
    }

    result = await parallel_read_stage(state, nodes)

    assert result["context"]["gssc_context"] == "ctx"
    assert result["parallel_read_results"]["rag_prepare"]["evidence"][0]["id"] == 1


@pytest.mark.asyncio
async def test_waiting_approval_is_preserved(monkeypatch):
    db = make_test_session()

    async def forbidden_context(state, nodes, payload):
        raise AssertionError("context should not run while waiting approval")

    async def forbidden_rag_prepare(state, payload):
        raise AssertionError("rag prepare should not run while waiting approval")

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", forbidden_context)
    monkeypatch.setattr(parallel_read_module, "_rag_prepare_branch", forbidden_rag_prepare)

    approval_payload = {"approval_id": "a1"}
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "tool",
        "status": "waiting_approval",
        "approval_payload": approval_payload,
        "pending_tool_call_id": 7,
        "final_payload": {"answer": "paused"},
        "final_output": "paused",
        "route_plan": {"intent": "tool.email", "route": ["tool_agent"], "risk_level": "L3"},
    }

    result = await parallel_read_stage(state, RuntimeNodes(db, {}))

    assert result["status"] == "waiting_approval"
    assert result["approval_payload"] is approval_payload
    assert result["pending_tool_call_id"] == 7
    assert result["final_payload"] == {"answer": "paused"}
    assert result["final_output"] == "paused"


def test_graph_wiring_contains_parallel_read_stage():
    graph_path = _ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_builder.py"
    text = graph_path.read_text(encoding="utf-8")

    assert 'workflow.add_edge("planner", "parallel_prefetch")' in text
    assert 'workflow.add_edge("parallel_prefetch", "parallel_read_stage")' in text
    assert 'workflow.add_edge("parallel_read_stage", "supervisor_observer")' in text
    assert 'workflow.add_conditional_edges(\n        "supervisor_observer"' in text
    assert 'workflow.add_edge("parallel_prefetch", "context_builder")' not in text
