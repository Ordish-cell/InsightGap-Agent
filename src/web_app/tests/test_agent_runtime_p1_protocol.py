import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import nodes as runtime_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.planner import plan_route
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.tests.db_test_utils import make_test_session


def test_execution_plan_wraps_route_plan_without_replacing_it():
    route_plan = {
        "intent": "rag",
        "route": ["rag_agent", "evaluator", "final_response"],
        "risk_level": "L1",
        "needs_approval": False,
        "expected_output": "answer_with_evidence",
        "answer_mode": "rag_qa",
    }

    execution_plan = execution_plan_from_route_plan(route_plan, {"run_id": 1})

    assert route_plan["route"] == ["rag_agent", "evaluator", "final_response"]
    assert execution_plan["intent"] == "rag"
    assert execution_plan["risk_level"] == "L1"
    assert execution_plan["needs_approval"] is False
    assert [task["agent"] for task in execution_plan["tasks"]] == route_plan["route"]
    assert [task["task_id"] for task in execution_plan["tasks"]] == [
        "0:rag_agent",
        "1:evaluator",
        "2:final_response",
    ]


@pytest.mark.asyncio
async def test_rag_agent_appends_agent_result(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes.rag_service, "ask", lambda *args, **kwargs: {
        "answer": "RAG answer",
        "answer_mode": "test",
        "evidence": [{"id": 1, "title": "Doc"}],
    })
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "use knowledge base",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "rag", "route": ["rag_agent"]}),
    }

    result = await RuntimeNodes(db, {}).rag_agent(state)

    assert result["rag_result"]["answer"] == "RAG answer"
    assert result["agent_results"][0]["agent"] == "rag_agent"
    assert result["agent_results"][0]["status"] == "ok"
    assert result["agent_results"][0]["evidence"]


@pytest.mark.asyncio
async def test_memory_agent_appends_failed_agent_result(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes.memory_service, "add_memory", lambda **kwargs: {
        "ok": False,
        "error": "vector store unavailable",
    })
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "remember that I prefer concise answers",
        "route_plan": {"intent": "memory", "route": ["memory_agent"], "risk_level": "L0"},
        "execution_plan": execution_plan_from_route_plan({"intent": "memory", "route": ["memory_agent"]}),
    }

    result = await RuntimeNodes(db, {}).memory_agent(state)

    assert result["memory_write_result"]["success"] is False
    assert result["agent_results"][0]["agent"] == "memory_agent"
    assert result["agent_results"][0]["status"] == "failed"
    assert "memory_write_failed" in result["agent_results"][0]["warnings"]


@pytest.mark.asyncio
async def test_evaluator_records_hard_constraints(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "route_plan": {"intent": "rag", "route": ["rag_agent", "evaluator"], "risk_level": "L1"},
        "rag_result": {"answer": "No evidence answer", "evidence": []},
        "memory_write_result": {"success": False, "error": "write failed"},
        "agent_results": [{"agent": "memory_agent", "status": "failed"}],
    }

    result = await RuntimeNodes(db, {}).evaluator(state)

    assert "evidence_missing" in result["final_warnings"]
    assert "memory_write_failed" in result["final_warnings"]
    assert result["evaluation_result"]["pass_"] is False
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_evaluator_preserves_waiting_approval(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "approval",
        "route_plan": {"intent": "tool", "route": ["tool_agent"], "risk_level": "L3"},
    }

    result = await RuntimeNodes(db, {}).evaluator(state)

    assert result["status"] == "waiting_approval"
    assert "approval_pending" in result["final_warnings"]


def test_planner_marks_explicit_research_and_project_diagnostics():
    explicit = plan_route("deep research AI infrastructure trends")
    diagnostic = plan_route("document upload failed, which modules should I inspect?")
    architecture = plan_route("explain the current architecture")

    assert explicit["explicit_research"] is True
    assert explicit["research_mode"] == "deep"
    assert diagnostic["explicit_research"] is False
    assert diagnostic["research_mode"] == "none"
    assert "research_agent" not in diagnostic["route"]
    assert architecture["explicit_research"] is False
    assert "research_agent" not in architecture["route"]


@pytest.mark.asyncio
async def test_research_agent_forces_fallback_when_not_explicit(monkeypatch):
    db = make_test_session()
    captured = {}

    async def fake_research_query(db_arg, user_id, request):
        captured["force_engine"] = request.force_engine
        return {"status": "completed", "summary": "fallback result", "findings": [], "evidence": []}

    monkeypatch.setattr(runtime_nodes.research_service, "research_query", fake_research_query)
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "document upload failed, which modules should I inspect?",
        "route_plan": {"intent": "research", "route": ["research_agent"], "explicit_research": False},
    }

    await RuntimeNodes(db, {}).research_agent(state)

    assert captured["force_engine"] == "fallback"


@pytest.mark.asyncio
async def test_research_agent_allows_odr_when_explicit(monkeypatch):
    db = make_test_session()
    captured = {}

    async def fake_research_query(db_arg, user_id, request):
        captured["force_engine"] = request.force_engine
        return {"status": "completed", "summary": "odr result", "findings": [], "evidence": []}

    monkeypatch.setattr(runtime_nodes.research_service, "research_query", fake_research_query)
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "deep research AI infrastructure trends",
        "route_plan": {"intent": "research", "route": ["research_agent"], "explicit_research": True},
    }

    await RuntimeNodes(db, {}).research_agent(state)

    assert captured["force_engine"] is None


@pytest.mark.asyncio
async def test_research_agent_appends_agent_result(monkeypatch):
    db = make_test_session()

    async def fake_research_query(db_arg, user_id, request):
        return {
            "status": "completed",
            "summary": "research summary",
            "findings": ["finding one"],
            "evidence": [{"url": "https://example.com"}],
            "metadata": {"engine": "fallback_researcher"},
        }

    monkeypatch.setattr(runtime_nodes.research_service, "research_query", fake_research_query)
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "research this",
        "route_plan": {"intent": "research", "route": ["research_agent"], "explicit_research": False},
        "execution_plan": execution_plan_from_route_plan({"intent": "research", "route": ["research_agent"]}),
    }

    result = await RuntimeNodes(db, {}).research_agent(state)

    research_result = result["agent_results"][0]
    assert research_result["agent"] == "research_agent"
    assert research_result["status"] == "ok"
    assert research_result["evidence"]
    assert "research_engine=fallback_researcher" in research_result["warnings"]


@pytest.mark.asyncio
async def test_artifact_agent_appends_agent_result(monkeypatch):
    db = make_test_session()

    monkeypatch.setattr(runtime_nodes.artifact_service, "save_text_artifact", lambda *args, **kwargs: "artifact.md")
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    class FakeArtifactRepository:
        def __init__(self, db_arg):
            pass

        def create(self, **kwargs):
            return SimpleNamespace(id=7, artifact_type=kwargs["artifact_type"], title=kwargs["title"], file_path=kwargs["file_path"])

    monkeypatch.setattr(runtime_nodes, "ArtifactRepository", FakeArtifactRepository)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "make artifact",
        "route_plan": {"intent": "artifact", "route": ["artifact_agent"], "risk_level": "L2"},
        "execution_plan": execution_plan_from_route_plan({"intent": "artifact", "route": ["artifact_agent"]}),
        "final_output": "content",
    }

    result = await RuntimeNodes(db, {}).artifact_agent(state)

    artifact_result = result["agent_results"][0]
    assert artifact_result["agent"] == "artifact_agent"
    assert artifact_result["status"] == "ok"
    assert artifact_result["artifacts"][0]["id"] == 7


@pytest.mark.asyncio
async def test_tool_agent_appends_agent_result(monkeypatch):
    db = make_test_session()

    monkeypatch.setattr(runtime_nodes, "infer_tool", lambda *args, **kwargs: ("local_file.read", {"path": "README.md"}))
    monkeypatch.setattr(runtime_nodes, "validate_tool_input", lambda tool_name, tool_input: (tool_input, []))
    monkeypatch.setattr(runtime_nodes.mcp_service, "call_tool", lambda *args, **kwargs: {
        "status": "completed",
        "tool_name": "local_file.read",
        "output": {"content": "hello"},
    })
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "read README",
        "route_plan": {"intent": "tool.local_file", "route": ["tool_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "tool.local_file", "route": ["tool_agent"]}),
    }

    result = await RuntimeNodes(db, {}).tool_agent(state)

    tool_result = result["agent_results"][0]
    assert tool_result["agent"] == "tool_agent"
    assert tool_result["status"] == "ok"
    assert tool_result["tool_calls"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_skill_agent_appends_agent_result(monkeypatch):
    db = make_test_session()

    monkeypatch.setattr(runtime_nodes.skill_service, "evaluate_reusability", lambda state: {
        "reusable_score": 0.2,
        "should_create": False,
        "reason": "low signal",
    })
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route_plan": {"intent": "skill", "route": ["skill_agent"], "risk_level": "L0"},
        "execution_plan": execution_plan_from_route_plan({"intent": "skill", "route": ["skill_agent"]}),
    }

    result = await RuntimeNodes(db, {}).skill_agent(state)

    skill_result = result["agent_results"][0]
    assert skill_result["agent"] == "skill_agent"
    assert skill_result["status"] == "ok"
    assert skill_result["confidence"] == 0.2


@pytest.mark.asyncio
async def test_final_response_appends_agent_result(monkeypatch):
    db = make_test_session()

    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route_plan": {"intent": "chat", "route": ["final_response"], "risk_level": "L0"},
        "execution_plan": execution_plan_from_route_plan({"intent": "chat", "route": ["final_response"]}),
        "final_output": "final answer",
    }

    result = await RuntimeNodes(db, {}).final_response(state)

    final_result = result["agent_results"][0]
    assert final_result["agent"] == "final_response"
    assert final_result["status"] == "ok"
    assert result["final_payload"]["agent_results"][0]["agent"] == "final_response"


@pytest.mark.asyncio
async def test_evaluator_records_tool_artifact_research_constraints(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "mixed",
        "route_plan": {
            "intent": "mixed",
            "route": ["research_agent", "artifact_agent", "tool_agent", "evaluator"],
            "risk_level": "L3",
            "explicit_research": False,
        },
        "tool_result": {"status": "failed", "error": "boom"},
        "artifact_result": {"error": "disk full"},
        "artifacts": [],
    }

    result = await RuntimeNodes(db, {}).evaluator(state)

    assert "artifact_missing" in result["final_warnings"]
    assert "tool_failed" in result["final_warnings"]
    assert "research_fallback_mode" in result["final_warnings"]
    assert result["evaluation_result"]["pass_"] is False
