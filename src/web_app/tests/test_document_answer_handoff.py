"""Selected document evidence reaches Supervisor without a second retrieval."""
from types import SimpleNamespace
from langchain_core.messages import AIMessageChunk
import pytest
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.agent.runtime.state import SupervisorAction
from src.web_app.services.rag_service import rag_service
from src.web_app.tests.test_chat_control import env as env
from src.web_app.tests.test_supervisor_loop import state
from src.web_app.tests.test_conversation_document_chat import add_file

@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_owner", [True, False])
async def test_scope_violation_stops_before_retrieval(env, monkeypatch, foreign_owner):
    from src.web_app.models.orm import Document, User
    document = add_file(env)
    def forbidden(*a, **k):
        pytest.fail("Document outside the authorized user/scope was retrieved")
    monkeypatch.setattr(rag_service, "ask", forbidden)
    with env.factory() as db:
        if foreign_owner:
            other = User(email="foreign-document@example.test", hashed_password="x")
            db.add(other)
            db.flush()
            db.get(Document, document).user_id = other.id
            db.commit()
        nodes = SupervisorNodes(db, {})
        s = state(env)
        await nodes.permission_guard(s)
        s["request"] = {"attachment_ids": [document]}
        s["current_action"] = {**SupervisorAction(action="rag", arguments={"query": "evidence", "document_ids": [document if foreign_owner else document + 1]}).model_dump(), "action_id": "scoped"}
        await nodes.capability(s)
    result = s["observations"][-1]
    assert result["status"] == "failed"
    assert "document_scope_unavailable" in result["error"] if foreign_owner else "document_scope_mismatch" in result["error"]

@pytest.mark.asyncio
@pytest.mark.parametrize("gssc", ["", "recent conversation"])
async def test_retrieval_context_always_reaches_answer_model(env, monkeypatch, gssc):
    document = add_file(env)
    calls, prompts = [], []
    def retrieve(user, query, **kwargs):
        calls.append(kwargs["document_ids"])
        return {"retrieval_status": "ok", "answer": "[E1] 实验说明",
                "evidence": [{"evidence_id": "E1", "source_title": "实验报告.docx",
                              "quote": "实验步骤：抓包并分析。", "document_id": document}]}
    async def stream(prompt):
        prompts.append(prompt)
        yield AIMessageChunk(content="实验报告介绍了抓包分析。[E1]")
    monkeypatch.setattr(rag_service, "ask", retrieve)
    monkeypatch.setattr("src.web_app.agent.runtime.nodes.get_chat_model", lambda *a, **k: SimpleNamespace(astream=stream, bind_tools=lambda tools: SimpleNamespace(astream=stream)))
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = state(env)
        await nodes.permission_guard(s)
        s.update(context={"gssc_context": gssc}, request={"attachment_ids": [document]},
                 current_action={**SupervisorAction(action="rag", arguments={"query": "步骤"}).model_dump(), "action_id": "read"})
        await nodes.capability(s)
        assert s["observations"][-1]["status"] == "ok", s["observations"]
        await nodes.supervisor(s)
    assert calls == [[document]]
    assert len(prompts) == 1
    assert "抓包并分析" in prompts[0][1].content
    assert "实验报告.docx" in prompts[0][1].content
    assert s["final_answer"].startswith("实验报告介绍")

@pytest.mark.asyncio
@pytest.mark.parametrize("retrieval_status", ["empty", "failed"])
async def test_empty_and_failed_retrieval_are_distinct_observations(env, monkeypatch, retrieval_status):
    monkeypatch.setattr(rag_service, "ask", lambda *a, **k: {"retrieval_status": retrieval_status, "evidence": []})
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = state(env)
        await nodes.permission_guard(s)
        s["current_action"] = {**SupervisorAction(action="rag", arguments={"query": "步骤"}).model_dump(), "action_id": "read"}
        await nodes.capability(s)
    assert s["observations"][0]["status"] == retrieval_status
    assert s["observations"][0]["retryable"] == (retrieval_status == "failed")
