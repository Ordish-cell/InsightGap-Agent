import pytest
from src.web_app.services.rag_service import rag_service, EvidenceBatch
from src.web_app.rag.evidence import ANSWER_CONTRACT
from src.web_app.tests.test_rag_hybrid_retrieval import hybrid_env
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.tests.test_agent_runtime_p3b_rag_prepare_empty_reuse import _patch_runtime_side_effects


def test_scoped_no_evidence_never_general_fallback(monkeypatch):
    monkeypatch.setattr(rag_service, "search", lambda *a, **kw: {"results": [], "retrieval_status": "empty"})
    result = rag_service.ask(1, "是什么？", document_ids=[42])
    assert not result["needs_general_fallback"]
    assert result["answer_mode"] == "no_evidence"


def test_failure_is_not_empty(monkeypatch):
    monkeypatch.setattr(rag_service, "search", lambda *a, **kw: {"results": [], "retrieval_status": "failed"})
    result = rag_service.ask(1, "是什么？", document_ids=[42])
    assert result["answer_mode"] == "retrieval_failed"
    assert not result["needs_general_fallback"]
    assert rag_service.search_evidence(1, "q").retrieval_status == "failed"


@pytest.mark.asyncio
async def test_scoped_successful_empty_prepare_is_reused(hybrid_env, monkeypatch):
    db, user, _ = hybrid_env
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr(rag_service, "ask", lambda *a, **kw: pytest.fail("duplicate retrieval"))
    state = {"user_id": user.id, "run_id": 1, "thread_id": "t", "user_input": "合同金额", "page_context": {"attachment_ids": [42]},
             "route_plan": {"intent": "document_qa", "route": ["rag_agent"]},
             "parallel_read_results": {"rag_prepare": {"status": "ok", "search_attempted": True, "evidence": [],
                 "query": "合同金额", "document_ids": [42], "retrieval_status": "empty"}}}
    result = await RuntimeNodes(db, {}).rag_agent(state)
    assert result["rag_result"]["_parallel_read_no_evidence_used"]
    assert not result["rag_result"]["needs_general_fallback"]


def test_contract_covers_conflict_and_partial_reading():
    assert "上传时间不代表生效时间" in ANSWER_CONTRACT
    assert "partial/retrieved" in ANSWER_CONTRACT


@pytest.mark.parametrize('gssc', ['', 'Prepared user context'])
def test_discarded_rag_draft_cannot_bypass_selected_evidence(gssc):
    state = {'user_input': 'q', 'context': {'gssc_context': gssc},
             'route_plan': {'intent': 'document_qa', 'route': ['rag_agent']},
             'rag_result': {'answer': 'DISCARDED_RAW_CANDIDATE',
                            'evidence': [{'quote': 'VISIBLE'}],
                            'context': {'document_context_block': '[E1] VISIBLE'}}}
    prompt = RuntimeNodes(None, {})._build_final_answer_prompt(state, 'DISCARDED_RAW_CANDIDATE')
    assert '[E1] VISIBLE' in prompt
    assert 'DISCARDED_RAW_CANDIDATE' not in prompt
