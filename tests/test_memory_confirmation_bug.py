"""Tests for the long-term memory false-confirmation bug fix.

These tests validate:
1. add_memory / add_with_dedup return structured save results
2. _save_extracted includes save_results
3. Planner routes tech stack declarations → memory (NOT rag)
4. Planner sets answer_mode=memory_confirm for declarations
5. _sanitize_memory_claims removes false claims
6. MEMORY_CONTEXT_POLICY constraints
"""

import sys
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Memory service return value tests
# ═══════════════════════════════════════════════════════════════════════════


def test_add_memory_returns_ok_and_qdrant_status():
    """add_memory() must return ok=True and qdrant_indexed status (no DB)."""
    import pytest
    try:
        from src.web_app.services.memory_service import memory_service
    except ImportError:
        pytest.skip("memory_service requires full environment (PostgreSQL, Qdrant)")

    result = memory_service.add_memory(
        user_id=1, content="测试记忆", memory_type="semantic",
        importance=0.9, metadata={"category": "tech_stack"},
    )
    assert result.get("ok") is True, f"Expected ok=True, got {result}"
    assert "qdrant_indexed" in result, f"Expected qdrant_indexed in result"
    assert "qdrant_point_id" in result
    assert "error" in result
    assert result.get("deduped") is False
    assert result.get("updated_existing") is False


def test_add_memory_includes_category_and_status():
    """add_memory() must include category and status from metadata."""
    import pytest
    try:
        from src.web_app.services.memory_service import memory_service
    except ImportError:
        pytest.skip("memory_service requires full environment (PostgreSQL, Qdrant)")

    result = memory_service.add_memory(
        user_id=1, content="偏好测试", memory_type="semantic",
        importance=0.85, metadata={"category": "preference", "status": "active"},
    )
    assert result.get("category") == "preference"
    assert result.get("status") == "active"


# ═══════════════════════════════════════════════════════════════════════════
# Planner routing tests
# ═══════════════════════════════════════════════════════════════════════════


def test_planner_tech_stack_to_memory_not_rag():
    """Tech stack declarations must NOT route to rag_agent even with Qdrant keyword."""
    from src.web_app.agent.runtime.planner import plan_route

    plan = plan_route("这个项目用FastAPI+PostgreSQL+Qdrant")
    assert "rag_agent" not in plan["route"], (
        f"Tech stack declaration should NOT route to rag, but got route={plan['route']}"
    )
    assert plan["intent"] in ("memory", "chat")


def test_planner_tech_stack_qdrant_suppress_rag():
    """'项目技术栈：Qdrant向量数据库' must suppress RAG even with 'Qdrant' keyword."""
    from src.web_app.agent.runtime.planner import plan_route

    plan = plan_route("项目技术栈：Qdrant向量数据库")
    assert "rag_agent" not in plan["route"], (
        f"Qdrant keyword should not trigger RAG for tech stack, got route={plan['route']}"
    )


def test_planner_name_preference_memory_confirm():
    """'我叫C' must route to memory with answer_mode=memory_confirm."""
    from src.web_app.agent.runtime.planner import plan_route

    plan = plan_route("我叫C，以后叫我C")
    assert plan.get("answer_mode") == "memory_confirm", (
        f"Expected memory_confirm, got {plan.get('answer_mode')}"
    )


def test_planner_greeting_is_casual():
    """Greetings must have answer_mode=casual or chat."""
    from src.web_app.agent.runtime.planner import plan_route

    plan = plan_route("你好")
    assert plan["intent"] == "chat"
    assert plan.get("answer_mode") in ("casual", "chat")


def test_planner_weather_is_chat_not_project_advice():
    """Weather question must NOT be project_advice."""
    from src.web_app.agent.runtime.planner import plan_route

    plan = plan_route("今天天气怎么样")
    assert plan["intent"] == "chat"
    assert plan.get("answer_mode") in ("casual", "chat"), (
        f"Weather should be casual/chat, got {plan.get('answer_mode')}"
    )


def test_planner_email_still_triggers():
    """Email intent must still work after tech-stack suppression changes."""
    from src.web_app.agent.runtime.planner import plan_route

    plan = plan_route("帮我发邮件给张三")
    assert str(plan["intent"]).startswith("tool.")
    assert plan["needs_approval"] is True


def test_planner_memory_write_is_memory_confirm():
    """Explicit memory write must have answer_mode=memory_confirm."""
    from src.web_app.agent.runtime.planner import plan_route

    plan = plan_route("记住我叫C")
    assert plan.get("answer_mode") == "memory_confirm"


# ═══════════════════════════════════════════════════════════════════════════
# Sanitize memory claims tests
# ═══════════════════════════════════════════════════════════════════════════


def test_sanitize_removes_false_claim_on_explicit_failure():
    """When memory_write_result.success=False, '已记住' must be stripped."""
    from src.web_app.agent.runtime.nodes import RuntimeNodes

    answer = "好的，已记住：你喜欢用FastAPI。还有其他需要吗？"
    save_results = [{"ok": False, "error": "db_error", "content": "test"}]
    mem_write = {"success": False, "error": "db_error"}

    result = RuntimeNodes._sanitize_memory_claims(answer, save_results, mem_write)
    assert "已记住" not in result, f"Should strip 已记住, got: {result}"
    assert "未能保存" in result, f"Should say 未能保存, got: {result}"


def test_sanitize_removes_false_claim_when_no_ok():
    """When no save_result has ok=True but save_results exists, must not claim success."""
    from src.web_app.agent.runtime.nodes import RuntimeNodes

    answer = "我记住了你的偏好"
    save_results = [{"ok": False, "error": "timeout"}, {"ok": False, "error": "db_error"}]
    mem_write = {}

    result = RuntimeNodes._sanitize_memory_claims(answer, save_results, mem_write)
    assert "没有确认写入成功" in result or "不能说已经记住" in result, (
        f"Must say not confirmed, got: {result}"
    )


def test_sanitize_passes_through_when_ok():
    """When save_results has ok=True, answer should pass through (with Qdrant note if needed)."""
    from src.web_app.agent.runtime.nodes import RuntimeNodes

    answer = "已记住：你喜欢用FastAPI。"
    save_results = [{"ok": True, "qdrant_indexed": True, "content": "test"}]
    mem_write = {"success": True}

    result = RuntimeNodes._sanitize_memory_claims(answer, save_results, mem_write)
    assert "已记住" in result, f"Should keep 已记住, got: {result}"
    assert "未能保存" not in result


def test_sanitize_qdrant_fail_adds_note():
    """When PG succeeded but Qdrant failed, answer should include note."""
    from src.web_app.agent.runtime.nodes import RuntimeNodes

    answer = "已记住：你喜欢用FastAPI。"
    save_results = [{"ok": True, "qdrant_indexed": False, "content": "test"}]
    mem_write = {"success": True}

    result = RuntimeNodes._sanitize_memory_claims(answer, save_results, mem_write)
    assert "已记住" in result, "Should keep 已记住"
    assert "向量索引暂不可用" in result, f"Should mention qdrant unavailability, got: {result}"


def test_sanitize_empty_save_results_no_effect():
    """When save_results is empty, answer should pass through unchanged."""
    from src.web_app.agent.runtime.nodes import RuntimeNodes

    answer = "你好，我是Agent OS助手。"
    result = RuntimeNodes._sanitize_memory_claims(answer, [], {})
    assert result == answer


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY_CONTEXT_POLICY tests
# ═══════════════════════════════════════════════════════════════════════════


def test_general_qa_disallows_tech_stack():
    """general_qa must NOT allow tech_stack, project_goal, boundary, workflow_pattern."""
    from src.web_app.context.builder import MEMORY_CONTEXT_POLICY

    allowed = MEMORY_CONTEXT_POLICY.get("general_qa", set())
    assert "project_goal" not in allowed, "general_qa must NOT allow project_goal"
    assert "tech_stack" not in allowed, "general_qa must NOT allow tech_stack"
    assert "boundary" not in allowed, "general_qa must NOT allow boundary"
    assert "workflow_pattern" not in allowed, "general_qa must NOT allow workflow_pattern"
    assert "name_preference" in allowed, "general_qa must allow name_preference"


def test_memory_confirm_allows_name_but_not_tech_stack():
    """memory_confirm must allow name/language/tone but NOT tech_stack/project_goal."""
    from src.web_app.context.builder import MEMORY_CONTEXT_POLICY

    allowed = MEMORY_CONTEXT_POLICY.get("memory_confirm", set())
    assert "name_preference" in allowed
    assert "language_preference" in allowed
    assert "tone_preference" in allowed
    assert "tech_stack" not in allowed, "memory_confirm must NOT allow tech_stack"
    assert "project_goal" not in allowed, "memory_confirm must NOT allow project_goal"


def test_casual_disallows_tech_stack():
    """casual mode must NOT inject tech_stack/project_goal."""
    from src.web_app.context.builder import MEMORY_CONTEXT_POLICY

    allowed = MEMORY_CONTEXT_POLICY.get("casual", set())
    assert "tech_stack" not in allowed
    assert "project_goal" not in allowed
    assert "name_preference" in allowed


def test_project_advice_allows_tech_stack():
    """project_advice must allow tech_stack and project_goal."""
    from src.web_app.context.builder import MEMORY_CONTEXT_POLICY

    allowed = MEMORY_CONTEXT_POLICY.get("project_advice", set())
    assert "tech_stack" in allowed
    assert "project_goal" in allowed
    assert "workflow_pattern" in allowed


# ═══════════════════════════════════════════════════════════════════════════
# State definition tests
# ═══════════════════════════════════════════════════════════════════════════


def test_state_has_memory_save_results_field():
    """AgentRuntimeState must have memory_save_results, memory_candidates, answer_mode."""
    from src.web_app.agent.runtime.state import AgentRuntimeState

    # Verify the TypedDict has the new fields (compile-time check)
    fields = AgentRuntimeState.__annotations__
    assert "memory_save_results" in fields, "Missing memory_save_results in state"
    assert "memory_candidates" in fields, "Missing memory_candidates in state"
    assert "answer_mode" in fields, "Missing answer_mode in state"


def test_route_plan_has_answer_mode():
    """RoutePlan must have answer_mode field."""
    from src.web_app.agent.runtime.state import RoutePlan

    fields = RoutePlan.__annotations__
    assert "answer_mode" in fields, "Missing answer_mode in RoutePlan"


def test_memory_save_result_typed_dict():
    """MemorySaveResult must have required fields."""
    from src.web_app.agent.runtime.state import MemorySaveResult

    fields = MemorySaveResult.__annotations__
    required_fields = {"ok", "memory_id", "qdrant_point_id", "memory_type", "content",
                       "category", "status", "qdrant_indexed", "error", "deduped", "updated_existing"}
    assert required_fields <= set(fields), f"Missing fields: {required_fields - set(fields)}"


# ═══════════════════════════════════════════════════════════════════════════
# Intent schema tests
# ═══════════════════════════════════════════════════════════════════════════


def test_home_intent_result_has_answer_mode():
    """HomeIntentResult must have answer_mode field."""
    from src.web_app.agent.runtime.intent_schema import HomeIntentResult

    result = HomeIntentResult(intent="chat", answer_mode="casual")
    assert result.answer_mode == "casual"

    d = result.to_home_intent_dict()
    assert "answer_mode" in d
    assert d["answer_mode"] == "casual"
