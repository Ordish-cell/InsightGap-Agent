import sys
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
async def test_rag_prepare_empty_search_records_attempt_and_count(monkeypatch):
    db = make_test_session()

    async def fake_context(state, nodes, payload):
        branch = dict(state)
        branch["context"] = {"gssc_context": "ctx"}
        return branch

    monkeypatch.setattr(parallel_read_module, "_context_skill_branch", fake_context)
    monkeypatch.setattr(parallel_read_module.rag_service, "search_evidence", lambda *args, **kwargs: [])

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route": "rag",
        "user_input": "unknown knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
    }

    result = await parallel_read_stage(state, RuntimeNodes(db, {}))

    prepared = result["parallel_read_results"]["rag_prepare"]
    assert prepared["status"] == "ok"
    assert prepared["evidence"] == []
    assert prepared["evidence_count"] == 0
    assert prepared["search_attempted"] is True


@pytest.mark.asyncio
async def test_rag_agent_reuses_empty_prepare_without_recalling_rag_ask(monkeypatch):
    import src.web_app.services.rag_service as rag_service_module

    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr(runtime_nodes.rag_service, "ask", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rag ask recalled")))

    async def fake_general_answer(query):
        return f"general answer for {query}"

    monkeypatch.setattr(rag_service_module, "_answer_from_general_llm", fake_general_answer)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "unknown knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "rag", "route": ["rag_agent"]}),
        "parallel_read_results": {
            "rag_prepare": {
                "status": "ok",
                "evidence": [],
                "evidence_count": 0,
                "search_attempted": True,
            }
        },
    }

    result = await RuntimeNodes(db, {}).rag_agent(state)

    assert result["rag_result"]["_parallel_read_no_evidence_used"] is True
    assert result["rag_result"]["_fallback_used"] is True
    assert result["rag_result"]["answer"] == "general answer for unknown knowledge question"
    assert result["rag_result"]["answer_mode"] == "general_knowledge_fallback"
    assert result["rag_result"]["evidence"] == []
    assert result["agent_results"][-1]["agent"] == "rag_agent"
    assert result["agent_results"][-1]["status"] == "ok"
    assert "evidence_missing" in result["agent_results"][-1]["warnings"]
    assert "rag_agent" in result["completed_nodes"]


@pytest.mark.asyncio
async def test_rag_prepare_missing_or_failed_still_uses_old_rag_logic(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    calls = []

    def fake_ask(*args, **kwargs):
        calls.append((args, kwargs))
        return {"answer": "formal answer", "evidence": [{"id": "formal"}]}

    monkeypatch.setattr(runtime_nodes.rag_service, "ask", fake_ask)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "rag", "route": ["rag_agent"]}),
        "parallel_read_results": {
            "rag_prepare": {
                "status": "failed",
                "evidence": [],
                "search_attempted": False,
            }
        },
    }

    result = await RuntimeNodes(db, {}).rag_agent(state)

    assert calls
    assert result["rag_result"]["answer"] == "formal answer"
    assert result["rag_result"]["evidence"][0]["id"] == "formal"


@pytest.mark.asyncio
async def test_attachment_document_query_does_not_use_empty_prepare_skip(monkeypatch):
    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    calls = []

    def fake_ask(*args, **kwargs):
        calls.append((args, kwargs))
        return {"answer": "attachment answer", "evidence": [{"id": "attachment"}]}

    monkeypatch.setattr(runtime_nodes.rag_service, "ask", fake_ask)

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
                "evidence": [],
                "evidence_count": 0,
                "search_attempted": True,
            }
        },
    }

    result = await RuntimeNodes(db, {"attachment_ids": [42]}).rag_agent(state)

    assert calls
    assert calls[0][1]["document_ids"] == [42]
    assert result["rag_result"]["answer"] == "attachment answer"
    assert not result["rag_result"].get("_parallel_read_no_evidence_used")


@pytest.mark.asyncio
async def test_document_qa_overview_uses_document_rag_instead_of_empty_prepare_skip(monkeypatch):
    import src.web_app.services.rag_service as rag_service_module

    db = make_test_session()
    _patch_runtime_side_effects(monkeypatch)
    calls = []

    monkeypatch.setattr(rag_service_module, "is_document_overview_query", lambda query: True)

    def fake_ask_document(*args, **kwargs):
        calls.append((args, kwargs))
        return {"answer": "document overview", "evidence": [{"id": "doc"}]}

    monkeypatch.setattr(runtime_nodes.rag_service, "ask_document", fake_ask_document)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "summary",
        "route_plan": {"intent": "document_qa", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "document_qa", "route": ["rag_agent"]}),
        "parallel_read_results": {
            "rag_prepare": {
                "status": "ok",
                "evidence": [],
                "evidence_count": 0,
                "search_attempted": True,
            }
        },
    }

    result = await RuntimeNodes(db, {"attachment_ids": [42]}).rag_agent(state)

    assert calls
    assert calls[0][1]["document_ids"] == [42]
    assert calls[0][1]["overview_mode"] is True
    assert result["rag_result"]["answer"] == "document overview"
    assert not result["rag_result"].get("_parallel_read_no_evidence_used")
