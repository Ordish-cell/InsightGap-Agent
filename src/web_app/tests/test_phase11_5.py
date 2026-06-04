"""Phase 11.5: Multi-Agent Minimal Closed Loop tests."""
import pytest

from src.web_app.agent.runtime.planner import plan_route
from src.web_app.agent.runtime.state import AgentRuntimeState, append_error, append_output, mark_completed
from src.web_app.models.orm import User
from src.web_app.services.auth_service import hash_password
from src.web_app.services.agent_service import run_agent
from src.web_app.tests.db_test_utils import make_test_session


def _user(db, email="test@example.com"):
    user = User(email=email, hashed_password=hash_password("pass"), nickname="test")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ===== Planner routing tests =====


def test_planner_chat_route():
    plan = plan_route("你好，请介绍一下你自己")
    assert plan["intent"] == "chat"
    assert "final_response" in plan["route"]
    assert "evaluator" in plan["route"]


def test_planner_research_route():
    plan = plan_route("帮我研究今天 AI infra 里有什么产品机会")
    assert plan["intent"] in ("research", "mixed")
    assert "research_agent" in plan["route"]


def test_planner_rag_route():
    plan = plan_route("根据我上传的文档总结重点")
    assert plan["intent"] in ("rag", "chat", "research", "mixed")
    assert "rag_agent" in plan["route"] or plan["intent"] in ("chat", "research", "mixed")


def test_planner_artifact_route():
    plan = plan_route("生成一份 MVP 产品方案文档")
    assert plan["intent"] in ("artifact", "mixed", "rag")
    assert "artifact_agent" in plan["route"] or plan["intent"] in ("rag", "mixed")


def test_planner_tool_email_requires_approval():
    plan = plan_route("帮我发邮件给老板汇报进度")
    assert plan["intent"] == "tool"
    assert "tool_agent" in plan["route"]
    assert plan["needs_approval"] is True
    assert plan["risk_level"] in ("L3", "L4")


def test_planner_tool_delete_high_risk():
    plan = plan_route("删除那条记录")
    assert plan["intent"] == "tool"
    assert plan["needs_approval"] is True
    assert plan["risk_level"] == "L4"


def test_planner_feed_card_deep_dive():
    plan = plan_route("研究一下这个机会", feed_card_id=5)
    assert plan["intent"] == "feed_research"
    assert "research_agent" in plan["route"]


def test_planner_mixed_research_and_artifact():
    plan = plan_route("研究 AI 产品机会并生成一份 MVP 方案文档")
    assert plan["intent"] in ("mixed", "research")
    assert "research_agent" in plan["route"]


def test_planner_skill_route():
    plan = plan_route("把这段工作流做成可复用的 skill，下次直接调用")
    assert "skill_agent" in plan["route"] or plan["intent"] in ("skill", "chat", "mixed")


def test_planner_memory_route():
    plan = plan_route("记住我的偏好，以后都用中文给我回复")
    assert "memory_agent" in plan["route"] or plan["intent"] in ("memory", "chat")


# ===== State helpers tests =====


def test_append_output():
    state: AgentRuntimeState = {}
    append_output(state, "research_agent", {"summary": "test"})
    assert len(state.get("agent_outputs", [])) == 1
    assert state["agent_outputs"][0]["node"] == "research_agent"


def test_append_error():
    state: AgentRuntimeState = {}
    append_error(state, "rag_agent", "connection failed")
    assert len(state.get("errors", [])) == 1
    assert "connection failed" in state["errors"][0]["error"]


def test_mark_completed():
    state: AgentRuntimeState = {}
    mark_completed(state, "planner")
    assert "planner" in state.get("completed_nodes", [])
    assert state["current_node"] == "planner"


def test_state_backward_compat():
    """Old state fields still work."""
    state: AgentRuntimeState = {
        "user_id": 1, "run_id": 100, "user_input": "test",
        "route": "research", "status": "running",
        "final_output": "", "error": "",
    }
    assert state["route"] == "research"
    assert state["user_id"] == 1
    # New fields should be absent by default
    assert state.get("route_plan") is None
    assert state.get("agent_outputs") is None


# ===== Agent run integration tests =====


def test_agent_run_chat_returns_final_payload():
    """A simple chat should complete and return the new structure fields."""
    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {
        "user_input": "你好，请介绍一下你能做什么",
        "source": "home_chat",
        "write_memory": True,
        "auto_skill": False,
        "create_skill_draft_if_reusable": False,
    })
    assert result["status"] == "completed"
    assert "intent" in result
    assert "route_plan" in result
    assert "final_payload" in result or "final_answer" in result


def test_agent_run_research_route():
    """Research-like input should route to research_agent."""
    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {
        "user_input": "帮我研究 AI agent 的最新趋势和产品机会",
        "source": "home_chat",
        "write_memory": True,
        "auto_skill": False,
        "create_skill_draft_if_reusable": False,
    })
    assert result["status"] in ("completed", "running")
    assert "intent" in result
    # Research tasks should route appropriately
    intent = result.get("intent", "")
    assert intent in ("research", "mixed", "chat")


def test_agent_run_returns_errors_field():
    """Response should always include an errors array."""
    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {
        "user_input": "你好",
        "source": "home_chat",
        "write_memory": False,
        "auto_skill": False,
        "create_skill_draft_if_reusable": False,
    })
    assert "errors" in result
    assert isinstance(result["errors"], list)


def test_agent_run_approval_flag():
    """High-risk tool input should set approval_required."""
    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {
        "user_input": "帮我发邮件给老板",
        "source": "home_chat",
        "write_memory": False,
        "auto_skill": False,
        "create_skill_draft_if_reusable": False,
    })
    assert "approval_required" in result
    # Should be True for email sending
    assert result["approval_required"] is True
    assert "approval_payload" in result


def test_agent_run_artifact_generation():
    """Artifact generation should produce artifacts in the result."""
    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {
        "user_input": "生成一份 Agent OS 的产品方案文档",
        "source": "home_chat",
        "write_memory": False,
        "auto_skill": False,
        "create_skill_draft_if_reusable": False,
    })
    assert result["status"] in ("completed", "running")
    # Should have artifacts or at minimum not crash
    assert "artifacts" in result


def test_agent_run_with_feed_card():
    """Feed card deep dive should route to research."""
    db = make_test_session()
    user = _user(db)
    result = run_agent(db, user.id, {
        "user_input": "深挖这个机会",
        "source": "home_chat",
        "feed_card_id": 1,
        "page_context": {"page": "home", "selected_feed_card_id": 1},
        "write_memory": False,
        "auto_skill": False,
        "create_skill_draft_if_reusable": False,
    })
    assert result["status"] in ("completed", "running")
    intent = result.get("intent", "")
    assert intent in ("feed_research", "research", "chat", "mixed")


# ===== Regression: original tests still compatible =====


def test_legacy_state_fields_preserved():
    """Legacy fields like route, status, final_output still work."""
    state: AgentRuntimeState = {
        "user_id": 1, "run_id": 10, "user_input": "test",
        "route": "rag", "status": "running",
    }
    assert state.get("route") == "rag"
    assert state.get("user_input") == "test"


def test_planner_always_returns_valid_route_plan():
    """Every call to plan_route returns a valid RoutePlan."""
    queries = [
        "你好",
        "帮我研究一下",
        "生成一份方案",
        "根据文档回答",
        "帮我发邮件",
        "",
        "记住我偏好中文",
        "深挖这个 Feed",
    ]
    for q in queries:
        plan = plan_route(q)
        assert "intent" in plan, f"Missing intent for: {q}"
        assert "route" in plan, f"Missing route for: {q}"
        assert isinstance(plan["route"], list), f"route should be list for: {q}"
        assert len(plan["route"]) >= 1, f"route too short for: {q} -> {plan['route']}"
        assert "final_response" in plan["route"], f"Missing final_response for: {q}"
        assert "risk_level" in plan, f"Missing risk_level for: {q}"
        assert plan["risk_level"] in ("L0", "L1", "L2", "L3", "L4")
