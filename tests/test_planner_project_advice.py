"""Tests for planner.py project_advice routing fix.

Verifies that tech-stack declarations + advice questions route to
project_advice (chat) and do NOT trigger the deep research pipeline.
"""

import sys
from pathlib import Path

# Ensure src/ is on the Python path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.web_app.agent.runtime.planner import plan_route


def _plan(user_input: str, **kwargs) -> dict:
    """Shorthand to call plan_route with defaults for a homepage chat scenario."""
    return plan_route(user_input=user_input, **kwargs)


# ── Test 1: tech-stack declaration + advice question → project_advice ──

def test_tech_stack_declaration_with_advice_question_routes_to_project_advice():
    """这个项目用 FastAPI + PostgreSQL + Qdrant 怎么设计架构？
    → answer_mode=project_advice, route does NOT contain research_agent."""
    plan = _plan("这个项目用 FastAPI + PostgreSQL + Qdrant 怎么设计架构？")
    assert plan["answer_mode"] == "project_advice", f"expected project_advice, got {plan['answer_mode']}"
    assert "research_agent" not in plan["route"], (
        f"research_agent should NOT be in route for project_advice, got {plan['route']}"
    )
    assert plan["intent"] in ("chat",), (
        f"intent should be chat for declaration+advice, got {plan['intent']}"
    )


# ── Test 2: pure tech-stack declaration → memory_confirm ──

def test_tech_stack_declaration_without_question_routes_to_memory_confirm():
    """这个项目用 FastAPI + PostgreSQL + Qdrant
    → answer_mode=memory_confirm, route contains memory_agent."""
    plan = _plan("这个项目用 FastAPI + PostgreSQL + Qdrant")
    assert plan["answer_mode"] == "memory_confirm", f"expected memory_confirm, got {plan['answer_mode']}"
    assert "memory_agent" in plan["route"], (
        f"memory_agent should be in route for memory_confirm, got {plan['route']}"
    )


# ── Test 3: explicit research request → research_agent ──

def test_explicit_research_request_routes_to_research_agent():
    """帮我调研 FastAPI + PostgreSQL + Qdrant 的生产最佳实践
    → route contains research_agent."""
    plan = _plan("帮我调研 FastAPI + PostgreSQL + Qdrant 的生产最佳实践")
    assert "research_agent" in plan["route"], (
        f"research_agent should be in route for explicit research, got {plan['route']}"
    )


def test_search_latest_routes_to_research_agent():
    """查一下 2026 年 Qdrant 最新部署方案
    → route contains research_agent."""
    plan = _plan("查一下 2026 年 Qdrant 最新部署方案")
    assert "research_agent" in plan["route"], (
        f"research_agent should be in route for latest info search, got {plan['route']}"
    )


# ── Test 4: general Q&A → chat ──

def test_general_qa_routes_to_chat():
    """帮我解释 FastAPI → general_qa/chat, no memory_agent, no research_agent."""
    plan = _plan("帮我解释 FastAPI")
    assert plan["intent"] in ("chat",), f"expected chat, got {plan['intent']}"
    assert "research_agent" not in plan["route"], (
        f"research_agent should NOT be in route for general Q&A, got {plan['route']}"
    )
    assert "memory_agent" not in plan["route"], (
        f"memory_agent should NOT be in route for general Q&A, got {plan['route']}"
    )


# ── Test 5: email/tool request → tool route ──

def test_email_request_route_unchanged():
    """帮我发邮件给张三说项目用 FastAPI → tool route with email intent."""
    plan = _plan("帮我发邮件给张三说项目用 FastAPI")
    assert plan["intent"] in ("tool.email", "tool"), f"expected tool intent, got {plan['intent']}"
    assert "tool_agent" in plan["route"], f"tool_agent should be in route, got {plan['route']}"
    assert "research_agent" not in plan["route"], (
        f"research_agent should NOT be in route for email request, got {plan['route']}"
    )


# ── Test 6: LLM intent research override is BLOCKED for declaration+advice ──

def test_llm_research_override_blocked_for_declaration_advice():
    """When LLM says research, but the input is declaration+advice, override is blocked."""
    plan = _plan(
        "这个项目用 FastAPI + PostgreSQL + Qdrant 怎么设计架构？",
        home_intent={"intent": "research", "confidence": 0.85, "risk_level": "L1"},
    )
    assert plan["answer_mode"] == "project_advice", f"expected project_advice, got {plan['answer_mode']}"
    assert "research_agent" not in plan["route"], (
        f"LLM research override should be blocked for declaration+advice, got {plan['route']}"
    )
    assert plan["intent"] in ("chat",), (
        f"intent should stay chat, got {plan['intent']}"
    )


# ── Test 7: explicit research overrides declaration+advice guard ──

def test_explicit_research_overrides_tech_stack_guard():
    """帮我调研 这个项目用 FastAPI + PostgreSQL + Qdrant 怎么设计架构
    → explicit research request wins → route contains research_agent."""
    plan = _plan("帮我调研 这个项目用 FastAPI + PostgreSQL + Qdrant 怎么设计架构")
    assert "research_agent" in plan["route"], (
        f"explicit research request should win, got {plan['route']}"
    )
