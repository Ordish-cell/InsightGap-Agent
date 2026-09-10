import json
import os
from time import perf_counter
from types import SimpleNamespace

import pytest
from langchain_core.messages import SystemMessage
from src.web_app.agent.prompts import GAP_BASE_PROMPT, gap_system_message
from src.web_app.agent.runtime.chat_fast_path import chat_entry
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.tests.test_chat_control import env
from src.web_app.tests.test_chat_fast_path import fake_model, initial
from src.web_app.tests.test_conversation_document_chat import add_file


def assert_identity(messages):
    systems = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(systems) == 1
    assert systems[0].content.count(GAP_BASE_PROMPT) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["chat", "clarify", "document"])
async def test_entry_and_document_identity_once(env, monkeypatch, route):
    did = add_file(env)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
    monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    decision = {"action": route, "document_ids": [did], "mode": "overview", "query": "总结文件"}
    calls, _, _ = fake_model(monkeypatch, [json.dumps(decision) + "\n", "回答或澄清"])
    answers = []
    async def answer(prompt):
        answers.append(prompt)
        yield SimpleNamespace(content="实验说明。")
    monkeypatch.setattr("src.web_app.agent.runtime.document_chat.get_chat_model", lambda *a, **k: SimpleNamespace(astream=answer))
    with env.factory() as db:
        result = await chat_entry(RuntimeNodes(db, {}), initial(env))
    assert len(calls) == 1
    assert_identity(calls[0])
    assert "FIRST line" in calls[0][0].content
    assert len(answers) == (1 if route == "document" else 0)
    if answers:
        assert_identity(answers[0])
        assert "coverage" in answers[0][0].content
    assert '"action"' not in result["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["legacy", "gssc", "recall"])
async def test_workflow_final_identity_and_continuation(env, monkeypatch, branch):
    from src.web_app.agent.runtime.node_groups import eval_final_nodes as mod
    calls = []
    async def stream(prompt):
        calls.append(prompt)
        yield SimpleNamespace(content="回答。")
    monkeypatch.setattr(mod, "get_chat_model", lambda *a, **k: SimpleNamespace(astream=stream))
    monkeypatch.setattr(mod, "get_llm_settings", lambda: SimpleNamespace(enabled=True))
    monkeypatch.setattr(mod, "resolve_model_name", lambda *a: SimpleNamespace(provider="fake", model="fake", tier="final"))
    state = initial(env)
    if branch == "gssc":
        state["context"] = {"gssc_context": "唯一上下文"}
    elif branch == "recall":
        state["answer_mode"] = "conversation_recall"
        state["conversation_recall_context"] = {"previous_user_messages": ["唯一历史"]}
    with env.factory() as db:
        answer = await RuntimeNodes(db, {"chat_continuation": "只说重点"})._generate_final_answer_with_llm(state, "任务结果")
    assert answer == "回答。" and len(calls) == 1
    assert_identity(calls[0])
    assert "只说重点" in calls[0][1].content
    assert "你是信息差 Agent OS" not in calls[0][1].content


@pytest.mark.asyncio
async def test_vision_answer_has_identity_internal_analysis_does_not(monkeypatch):
    from src.web_app.services.qwen_multimodal_service import QwenMultimodalService
    calls = []
    async def invoke(messages):
        calls.append(messages)
        return SimpleNamespace(content="图片内容")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: SimpleNamespace(ainvoke=invoke))
    monkeypatch.setattr("src.web_app.services.qwen_multimodal_service.settings.qwen_vision_api_key", "test-only")
    service = QwenMultimodalService()
    monkeypatch.setattr(service, "_build_image_message_content", lambda prompt, images: [{"type": "text", "text": prompt}])
    images = [{"filename": "test.png"}]
    await service.answer_image_question("是什么", images)
    await service.analyze_images("是什么", images)
    assert len(calls) == 2
    assert_identity(calls[0])
    assert GAP_BASE_PROMPT not in calls[1][0].content


def test_internal_prompts_remain_separate():
    from src.web_app.agent.runtime.intent_llm import _build_prompt
    from src.web_app.agent.runtime.llm_supervisor_prompts import WEB_APP_LLM_SUPERVISOR_SYSTEM_PROMPT
    from src.web_app.services.conversation_summary_service import CONVERSATION_SUMMARY_UPDATE_PROMPT, SEGMENT_CREATION_PROMPT
    assert GAP_BASE_PROMPT not in _build_prompt("你好", {}, None, "")
    assert GAP_BASE_PROMPT not in WEB_APP_LLM_SUPERVISOR_SYSTEM_PROMPT
    assert GAP_BASE_PROMPT not in CONVERSATION_SUMMARY_UPDATE_PROMPT
    assert GAP_BASE_PROMPT not in SEGMENT_CREATION_PROMPT
    assert len(gap_system_message().content) < 700


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("GAP_IDENTITY_SMOKE") != "1", reason="opt-in isolated provider smoke")
@pytest.mark.parametrize("scenario", ["identity", "casual", "explanation", "document", "workflow", "model"])
async def test_real_identity_smoke(env, monkeypatch, scenario):
    from sqlalchemy import select, text
    from src.web_app.db.session import SessionLocal
    from src.web_app.models.orm import AgentRun
    from src.web_app.services.llm_registry_service import resolve_run_model_context
    from src.web_app.agent.llm.context import use_model_context
    with SessionLocal() as source:
        source.execute(text("SET TRANSACTION READ ONLY"))
        run = source.scalar(select(AgentRun).where(AgentRun.status == "completed").order_by(AgentRun.id.desc()))
        if not run:
            pytest.skip("No configured model")
        ctx = resolve_run_model_context(source, run.user_id, (run.graph_state or {}).get("model_context") or {})
    questions = {"identity": "你是谁？", "casual": "唉，我好累", "explanation": "解释一下 RAG",
                 "document": "刚刚上传的实验报告讲什么？", "workflow": "你是谁？再简要说明资料中的实验步骤。",
                 "model": "你是什么底层模型，哪家供应商的？不知道就直说。"}
    if scenario == "document":
        add_file(env)
        monkeypatch.setattr("src.web_app.core.config.settings.chat_document_path_enabled", True)
        monkeypatch.setattr("src.web_app.services.document_chat_reader.SessionLocal", env.factory)
    monkeypatch.setattr("src.web_app.core.config.settings.chat_fast_path_enabled", True)
    started = perf_counter()
    with use_model_context(ctx), env.factory() as db:
        nodes = RuntimeNodes(db, {})
        state = {**initial(env), "user_input": questions[scenario]}
        if scenario == "workflow":
            state["context"] = {"gssc_context": "已取得实验资料：步骤为抓包，再分析网络协议。"}
            answer = await nodes._generate_final_answer_with_llm(state, "步骤为抓包，再分析网络协议。")
        else:
            result = await chat_entry(nodes, state)
            assert result["chat_entry_route"] == "chat"
            answer = result["final_answer"]
    assert answer.strip()
    if scenario in {"identity", "workflow"}:
        assert "Gap" in answer and "信息差" in answer
    assert "最终回复节点" not in answer
    if scenario == "document":
        assert "协议" in answer or "抓包" in answer
    print("GAP_SMOKE " + json.dumps({"scenario": scenario, "model": ctx.model,
        "elapsed_ms": round((perf_counter() - started) * 1000), "answer": answer}, ensure_ascii=False))
