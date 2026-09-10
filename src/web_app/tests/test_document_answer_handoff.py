from types import SimpleNamespace

import pytest

from src.web_app.agent.runtime import live_progress
from src.web_app.agent.runtime.llm_supervisor import _intent_for_decision
from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.services.rag_service import RAGService as RagService
from src.web_app.tests.db_test_utils import make_test_session
from src.web_app.tests.test_agent_runtime_p7d_final_response_node_result import _base_state, _patch_final_side_effects


def prepared_evidence():
    return [{"document_id": 57, "chunk_id": "child-1", "parent_id": "parent-1",
             "content": "实验目的：验证网络协议。", "parent_context": "实验目的：验证网络协议。实验步骤：抓包并分析。",
             "source_name": "实验报告.docx", "source_url": "", "score": 0.8, "metadata": {}}]


def test_prefetch_evidence_retains_text_and_source():
    service = RagService.__new__(RagService)
    evidence = service._evidence_from_results(prepared_evidence())
    assert evidence[0]["quote"] == prepared_evidence()[0]["parent_context"]
    assert evidence[0]["source_title"] == "实验报告.docx"
    assert "抓包并分析" in service._document_context_block(evidence)


def test_supervisor_preserves_document_qa():
    decision = SimpleNamespace(target_runtime="local", route=["rag_agent", "final_response"])
    assert _intent_for_decision({"route_plan": {"intent": "document_qa"}}, decision) == "document_qa"
    assert _intent_for_decision({"route_plan": {"intent": "chat"}}, decision) == "rag"


@pytest.mark.asyncio
async def test_rag_node_reuses_actual_prefetch_shape(monkeypatch):
    from src.web_app.agent.runtime import nodes as runtime_nodes
    from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
    from src.web_app.tests.test_agent_runtime_p3b_rag_prepare_empty_reuse import _patch_runtime_side_effects
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr("src.web_app.agent.runtime.node_groups.agent_nodes.resolve_model_name", lambda *a, **kw: SimpleNamespace(model="fake-model"))
    monkeypatch.setattr(runtime_nodes.rag_service, "ask", lambda *a, **kw: pytest.fail("repeated retrieval"))
    state = _base_state(final_output="")
    state["user_input"] = "实验步骤是什么？"
    state["route_plan"] = {"intent": "document_qa", "route": ["rag_agent"], "risk_level": "L1"}
    state["execution_plan"] = execution_plan_from_route_plan(state["route_plan"])
    state["parallel_read_results"] = {"rag_prepare": {"status": "ok", "evidence": prepared_evidence()}}
    db = make_test_session()
    try:
        result = await RuntimeNodes(db, {"attachment_ids": [57]}).rag_agent(state)
        rag = result["rag_result"]
        assert rag.get("answer_mode") == "retrieval_context", result.get("errors")
        assert "抓包并分析" in rag["context"]["document_context_block"]
        assert rag["evidence"][0]["source_title"] == "实验报告.docx"
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gssc", ["", "recent conversation"])
async def test_retrieval_context_always_reaches_answer_model(monkeypatch, gssc):
    _patch_final_side_effects(monkeypatch)
    service = RagService.__new__(RagService)
    evidence = service._evidence_from_results(prepared_evidence())
    state = _base_state(final_output="")
    state.update(user_input="这个文档有啥内容？", route_plan={"intent": "rag", "route": ["rag_agent", "final_response"]},
                 context={"gssc_context": gssc}, rag_result={"answer": "[document_qa_context]", "answer_mode": "retrieval_context",
                 "evidence": evidence, "context": {"document_context_block": service._document_context_block(evidence)}})
    calls = []

    async def generate(self, received, draft):
        prompt = self._build_final_answer_prompt(received, draft)
        assert "抓包并分析" in prompt
        assert "实验报告.docx" in prompt
        assert "[document_qa_context]" not in draft
        calls.append(prompt)
        return "这份实验报告介绍了通过抓包分析验证网络协议的实验。"

    monkeypatch.setattr(eval_final_nodes.EvalFinalNodesMixin, "_generate_final_answer_with_llm", generate)
    db = make_test_session()
    try:
        result = await RuntimeNodes(db, {}).final_response(state)
        assert len(calls) == 1
        assert "实验报告介绍" in result["final_answer"]
    finally:
        db.close()


def test_progress_reports_retrieval_fact_not_internal_prompt(monkeypatch):
    events = []
    monkeypatch.setattr("src.web_app.agent.runtime.event_ledger.publish_event", lambda *args, **kwargs: events.append(args[4]))
    live_progress.publish_findings(None, {"run_id": 1, "rag_result": {
        "answer": "以下是从当前上传文档中检索到的相关内容，请基于这些内容回答：",
        "evidence": prepared_evidence()}}, "rag_agent", "step1")
    assert len(events) == 1
    assert "实验报告.docx" in events[0]["text"]
    assert "1 段" in events[0]["text"]
    assert "请基于" not in events[0]["text"]
    assert events[0]["references"][0]["document_id"] == 57


def test_failed_generation_does_not_echo_internal_prompt():
    db = make_test_session()
    try:
        result = RuntimeNodes(db, {})._fallback_final_answer({"rag_result": {"answer_mode": "extractive_fallback"}}, "请基于这些内容回答：")
        assert "请重试" in result
        assert "请基于" not in result
    finally:
        db.close()


@pytest.mark.asyncio
async def test_document_overview_is_not_replaced_by_prefetched_top_k(monkeypatch):
    from src.web_app.agent.runtime import nodes as runtime_nodes
    from src.web_app.tests.test_agent_runtime_p3b_rag_prepare_empty_reuse import _patch_runtime_side_effects
    _patch_runtime_side_effects(monkeypatch)
    monkeypatch.setattr("src.web_app.agent.runtime.node_groups.agent_nodes.resolve_model_name", lambda *a, **kw: SimpleNamespace(model="fake-model"))
    monkeypatch.setattr("src.web_app.services.rag_service.is_document_overview_query", lambda _: True)
    calls = []
    def overview(user_id, question, **kwargs):
        calls.append(kwargs)
        return {"answer": "[document_qa_context]", "answer_mode": "document_overview_fallback", "evidence": prepared_evidence()}
    monkeypatch.setattr(runtime_nodes.rag_service, "ask_document", overview)
    state = _base_state("")
    state["route_plan"] = {"intent": "document_qa", "route": ["rag_agent"]}
    state["parallel_read_results"] = {"rag_prepare": {"status": "ok", "evidence": prepared_evidence()}}
    db = make_test_session()
    try:
        result = await RuntimeNodes(db, {"attachment_ids": [57]}).rag_agent(state)
        assert result["rag_result"]["answer_mode"] == "document_overview_fallback"
        assert len(calls) == 1
        assert calls[0]["document_ids"] == [57]
        assert calls[0]["overview_mode"] is True
    finally:
        db.close()
