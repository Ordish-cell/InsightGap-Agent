import asyncio
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import nodes as runtime_nodes
from src.web_app.agent.runtime import prefetch as prefetch_module
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.prefetch import parallel_prefetch
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.tests.db_test_utils import make_test_session


@pytest.mark.asyncio
async def test_parallel_prefetch_collects_results(monkeypatch):
    db = make_test_session()

    async def fake_rag(user_id, query):
        return {"evidence": [{"id": 1}], "count": 1}

    async def fake_memory(user_id, query, db_arg):
        return {"items": [{"id": 2}], "count": 1, "backend": "test"}

    async def fake_skill(user_id, query, db_arg, state, payload):
        return {"matched_skill": {"id": 3}, "candidate_skills": [{"id": 3}]}

    async def fake_graph(user_id, query, route):
        return {"context": "graph facts", "debug": {"ok": True}}

    monkeypatch.setattr(prefetch_module, "_rag_prefetch", fake_rag)
    monkeypatch.setattr(prefetch_module, "_memory_prefetch", fake_memory)
    monkeypatch.setattr(prefetch_module, "_skill_prefetch", fake_skill)
    monkeypatch.setattr(prefetch_module, "_graph_prefetch", fake_graph)

    state = {"user_id": 1, "run_id": 1, "thread_id": "t", "user_input": "hello", "route": "chat"}

    result = await parallel_prefetch(state, db, {})

    assert result["prefetch_results"]["rag"]["count"] == 1
    assert result["prefetch_results"]["memory"]["backend"] == "test"
    assert result["prefetch_results"]["skill"]["matched_skill"]["id"] == 3
    assert result["prefetch_results"]["graph"]["context"] == "graph facts"
    assert result["prefetch_elapsed_ms"] >= 0
    assert len(result["prefetch_agent_results"]) == 4
    assert any(item["agent"] == "rag_prefetch" for item in result["agent_results"])


@pytest.mark.asyncio
async def test_parallel_prefetch_failure_and_timeout_do_not_block(monkeypatch):
    db = make_test_session()

    async def slow_rag(user_id, query):
        await asyncio.sleep(0.05)
        return {"evidence": [{"id": 1}], "count": 1}

    async def failed_memory(user_id, query, db_arg):
        raise RuntimeError("memory down")

    async def ok_skill(user_id, query, db_arg, state, payload):
        return {"matched_skill": None, "candidate_skills": []}

    async def ok_graph(user_id, query, route):
        return {"context": "", "debug": {}}

    monkeypatch.setattr(prefetch_module, "_rag_prefetch", slow_rag)
    monkeypatch.setattr(prefetch_module, "_memory_prefetch", failed_memory)
    monkeypatch.setattr(prefetch_module, "_skill_prefetch", ok_skill)
    monkeypatch.setattr(prefetch_module, "_graph_prefetch", ok_graph)

    state = {"user_id": 1, "run_id": 1, "thread_id": "t", "user_input": "hello", "route": "chat"}

    result = await parallel_prefetch(state, db, {}, timeout_seconds=0.01)

    assert "prefetch_timeout:rag" in result["prefetch_warnings"]
    assert any(warning.startswith("memory_prefetch_failed") for warning in result["prefetch_warnings"])
    assert result["prefetch_results"]["skill"]["candidate_skills"] == []


@pytest.mark.asyncio
async def test_context_builder_reuses_prefetch_without_recalling_services(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes.memory_service, "search_memory", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("memory recalled")))
    monkeypatch.setattr(runtime_nodes.rag_service, "search_evidence", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rag recalled")))
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes.user_growth_service, "build_dynamic_preference_profile", lambda *args, **kwargs: {"preference_summary": ""})

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "route": "chat",
        "route_plan": {"intent": "chat", "answer_mode": "chat"},
        "prefetch_results": {
            "memory": {"items": [{"content": "pref memory", "metadata": {"category": "answer_preference"}}], "backend": "prefetch", "qdrant_hits": 1},
            "rag": {"evidence": [{"title": "pref doc", "content": "pref evidence"}]},
            "graph": {"context": "pref graph", "debug": {"source": "prefetch"}},
        },
    }

    result = await RuntimeNodes(db, {}).context_builder(state)

    assert result["context"]["memory_items"][0]["content"] == "pref memory"
    assert result["context"]["rag_evidence"][0]["title"] == "pref doc"
    assert result["context"]["graph_context"] == "pref graph"


@pytest.mark.asyncio
async def test_skill_matcher_reuses_prefetch_without_recalling_service(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes.skill_service, "match_skill", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("skill recalled")))
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "hello",
        "context": {"gssc_context": ""},
        "prefetch_results": {
            "skill": {
                "matched_skill": {"id": 9, "match_score": 0.91},
                "candidate_skills": [{"id": 9, "match_score": 0.91}],
            }
        },
    }

    result = await RuntimeNodes(db, {}).skill_matcher(state)

    assert result["matched_skill"]["id"] == 9
    assert result["candidate_skills"][0]["id"] == 9


@pytest.mark.asyncio
async def test_prefetch_timeout_still_allows_rag_agent_old_logic(monkeypatch):
    db = make_test_session()
    monkeypatch.setattr(runtime_nodes.rag_service, "ask", lambda *args, **kwargs: {
        "answer": "late rag answer",
        "evidence": [{"id": "formal"}],
    })
    monkeypatch.setattr(runtime_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_nodes, "record_event", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "knowledge question",
        "route_plan": {"intent": "rag", "route": ["rag_agent"], "risk_level": "L1"},
        "execution_plan": execution_plan_from_route_plan({"intent": "rag", "route": ["rag_agent"]}),
        "prefetch_warnings": ["prefetch_timeout:rag"],
        "prefetch_results": {},
    }

    result = await RuntimeNodes(db, {}).rag_agent(state)

    assert result["rag_result"]["answer"] == "late rag answer"
    assert result["rag_result"]["evidence"][0]["id"] == "formal"


def test_graph_wiring_contains_parallel_prefetch():
    graph_path = _ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_builder.py"
    text = graph_path.read_text(encoding="utf-8")

    assert 'workflow.add_edge("planner", "parallel_prefetch")' in text
    assert 'workflow.add_edge("parallel_prefetch", "parallel_read_stage")' in text
