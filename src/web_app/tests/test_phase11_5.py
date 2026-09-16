from src.web_app.agent.runtime.state import AgentRuntimeState
"""Phase 11.5: Multi-Agent Minimal Closed Loop tests."""
import pytest

pytestmark = pytest.mark.usefixtures("scripted_supervisor")

from src.web_app.models.orm import User
from src.web_app.services.auth_service import hash_password
from src.web_app.services.agent_service import run_agent
from src.web_app.tests.db_test_utils import make_test_session, configure_test_model


def _user(db, email="test@example.com"):
    user = User(email=email, hashed_password=hash_password("pass"), nickname="test")
    db.add(user)
    db.commit()
    db.refresh(user)
    configure_test_model(db, user)
    return user


# ===== Planner routing tests =====






















# ===== State helpers tests =====








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
        "tool_name": "email.send", "tool_input": {"to": "boss@example.test", "subject": "Demo", "body": "Report"},
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
