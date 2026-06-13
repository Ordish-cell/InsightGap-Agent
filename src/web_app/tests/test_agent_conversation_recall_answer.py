import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.node_groups import read_nodes
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.planner import plan_route
from src.web_app.services.agent_service import build_user_facing_answer
from src.web_app.tests.db_test_utils import make_test_session


def _recall_state() -> dict:
    previous_questions = [
        "现在项目的整体信息给我输出一遍，架构清晰",
        "现在教我怎么手动测试",
    ]
    return {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "conversation_id": "c",
        "user_input": "我前面问过你啥？分析一下我的意图",
        "route": "chat",
        "route_plan": {
            "intent": "chat",
            "route": ["evaluator", "final_response"],
            "risk_level": "L0",
            "answer_mode": "conversation_recall",
        },
        "completed_nodes": [],
        "agent_results": [],
        "agent_outputs": [],
        "errors": [],
        "final_output": "",
        "visible_thoughts": [],
        "langgraphstatus": {},
        "conversation_recall_context": {
            "source": "AgentConversation/AgentMessage",
            "previous_user_messages": previous_questions,
            "messages": [
                {"role": "user", "content": previous_questions[0]},
                {"role": "assistant", "content": "已输出项目架构。"},
                {"role": "user", "content": previous_questions[1]},
                {"role": "assistant", "content": "已给出手动测试步骤。"},
                {"role": "user", "content": "我前面问过你啥？分析一下我的意图"},
            ],
        },
        "context": {
            "conversation_history": (
                f"User: {previous_questions[0]}\n\n"
                "Assistant: 已输出项目架构。\n\n"
                f"User: {previous_questions[1]}\n\n"
                "Assistant: 已给出手动测试步骤。\n\n"
                "User: 我前面问过你啥？分析一下我的意图"
            ),
            "memory_items": [
                {"content": "这条长期记忆不能用于 conversation_recall", "memory_type": "semantic"}
            ],
        },
        "memory_context": {
            "skipped": True,
            "items": [],
            "skip_reason": "conversation_recall_uses_conversation_history_only",
        },
    }


def test_planner_routes_recall_without_memory_writer_or_memory_loader():
    plan = plan_route("我前面问过你啥？分析一下我的意图")

    assert plan["intent"] == "chat"
    assert plan["answer_mode"] == "conversation_recall"
    assert "memory_agent" not in plan["route"]
    assert "rag_agent" not in plan["route"]
    assert "research_agent" not in plan["route"]
    assert plan["memory_context_loader"] is False
    assert plan["memory_writer_planned"] is False


def test_build_user_facing_answer_does_not_template_override_llm_answer():
    state = _recall_state()
    state["final_payload"] = {"answer": "LLM 基于会话历史生成的答案。"}
    state["final_answer"] = "LLM 基于会话历史生成的答案。"

    assert build_user_facing_answer(state) == "LLM 基于会话历史生成的答案。"


@pytest.mark.asyncio
async def test_runtime_final_response_calls_llm_for_conversation_recall(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))

    calls: list[dict] = []

    async def fake_llm(self, state, draft_answer):
        calls.append({"state": state, "draft_answer": draft_answer})
        return "你前面问过项目架构和手动测试；你的意图是在确认我能否读取当前会话历史。"

    monkeypatch.setattr(RuntimeNodes, "_generate_final_answer_with_llm", fake_llm)

    result = await RuntimeNodes(make_test_session(), {}).final_response(_recall_state())

    assert len(calls) == 1
    assert "项目架构" in result["final_answer"]
    assert result["final_payload"]["conversation_recall_context"]["previous_user_messages"]
    assert result["final_payload"]["memory_context"]["skipped"] is True
    assert "load_memory_context" not in [step["key"] for step in result["final_payload"]["pipeline_steps"]]


@pytest.mark.asyncio
async def test_runtime_final_response_retries_no_history_claim(monkeypatch):
    monkeypatch.setattr(eval_final_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "emit_visible_thought", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_final_nodes, "resolve_model_name", lambda *args, **kwargs: SimpleNamespace(model="test-model"))

    responses = [
        "我没有历史记录，每次对话都是独立的。",
        "你前面问过项目架构和手动测试。",
    ]

    async def fake_llm(self, state, draft_answer):
        return responses.pop(0)

    monkeypatch.setattr(RuntimeNodes, "_generate_final_answer_with_llm", fake_llm)

    result = await RuntimeNodes(make_test_session(), {}).final_response(_recall_state())

    assert result["final_answer"] == "你前面问过项目架构和手动测试。"
    assert result["_conversation_recall_retry"] is True


@pytest.mark.asyncio
async def test_context_builder_skips_long_term_memory_for_conversation_recall(monkeypatch):
    def fail_memory_search(*args, **kwargs):
        raise AssertionError("conversation_recall must not read long-term memory")

    monkeypatch.setattr(read_nodes.memory_service, "search_memory", fail_memory_search)
    monkeypatch.setattr(read_nodes.memory_service, "get_baseline_memories", fail_memory_search)
    monkeypatch.setattr(read_nodes.rag_service, "search_evidence", lambda *args, **kwargs: [])
    monkeypatch.setattr(read_nodes.user_growth_service, "build_dynamic_preference_profile", lambda *args, **kwargs: {})
    monkeypatch.setattr(read_nodes, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(read_nodes, "append_status_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(read_nodes, "emit_visible_thought", lambda *args, **kwargs: None)

    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "conversation_id": "missing",
        "user_input": "我前面问过你啥？",
        "route": "chat",
        "route_plan": {"intent": "chat", "answer_mode": "conversation_recall", "route": ["evaluator", "final_response"]},
        "prefetch_results": {},
        "context": {},
    }

    result = await RuntimeNodes(make_test_session(), {}).context_builder(state)

    assert result["memory_context"]["skipped"] is True
    assert result["memory_context"]["items"] == []
    assert "load_memory_context" not in [step["key"] for step in result["pipeline_steps"]]


@pytest.mark.asyncio
async def test_memory_writer_outputs_skipped_decision_for_plain_chat(monkeypatch):
    monkeypatch.setattr(read_nodes, "record_step", lambda *args, **kwargs: None)
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "user_input": "你好",
        "route": "chat",
        "route_plan": {"intent": "chat", "route": ["final_response"]},
        "completed_nodes": [],
        "agent_results": [],
    }

    result = await RuntimeNodes(make_test_session(), {}).memory_agent(state)

    assert result["memory_write_decision"]["should_write"] is False
    assert result["memory_write_decision"]["mode"] == "skipped"
    assert result["memory_write_decision"]["reasons"] == ["memory_writer_not_requested"]
