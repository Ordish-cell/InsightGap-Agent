import pytest
from src.web_app.services.rag_service import rag_service, EvidenceBatch
from src.web_app.rag.evidence import ANSWER_CONTRACT
from src.web_app.tests.test_rag_hybrid_retrieval import hybrid_env
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.agent.runtime.context import bounded_prompt
from src.web_app.tests.test_chat_control import env as env
from src.web_app.tests.test_supervisor_loop import state as initial
from src.web_app.agent.runtime.state import CapabilityResult


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
async def test_completed_empty_action_is_not_retrieved_again(env, monkeypatch):
    monkeypatch.setattr(rag_service, "ask", lambda *a, **kw: pytest.fail("duplicate retrieval"))
    with env.factory() as db:
        s = initial(env)
        s["current_action"] = {"action": "rag", "action_id": "read", "arguments": {"query": "合同金额"}}
        s["observations"] = [CapabilityResult(action_id="read", capability="rag", status="empty").model_dump()]
        result = await SupervisorNodes(db, {}).capability(s)
    assert len(result["observations"]) == 1
    assert result["observations"][0]["status"] == "empty"


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
    state['observations'] = [CapabilityResult(action_id='read', capability='rag', status='ok', summary='[E1] VISIBLE').model_dump()]
    prompt = bounded_prompt(state, 'Use selected evidence', 16000)
    assert '[E1] VISIBLE' in prompt
    assert 'DISCARDED_RAW_CANDIDATE' not in prompt
