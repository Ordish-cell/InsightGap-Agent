"""Memory receipts, confirmation quality policy, and context isolation."""
import pytest
from types import SimpleNamespace

@pytest.mark.parametrize("claim,receipts,expected", [
    ("已记住你的偏好", [{"ok": False, "error": "db_error"}], "没有确认写入成功"),
    ("我记住了你的偏好", [{"ok": False, "error": "timeout"}], "没有确认写入成功"),
    ("已记住你的偏好", [{"ok": True, "qdrant_indexed": True}], ""),
    ("已记住你的偏好", [{"ok": True, "qdrant_indexed": False}], "向量索引暂不可用"),
    ("你好", [], ""),
    ("已记住你的偏好", [], "没有确认写入成功"),
    ("偏好已保存，我记住了", [{"ok": False}], "没有确认写入成功"),
])
def test_memory_claim_requires_receipt(claim, receipts, expected):
    from src.web_app.agent.runtime.policy import check_answer
    state = {"memory_save_results": receipts}
    correction = check_answer(SimpleNamespace(db=None), state, claim)
    if expected:
        assert expected in correction
        assert state["evaluation"]["warnings"]
    else:
        assert correction == ""

def test_state_keeps_memory_receipts_and_observations():
    from src.web_app.agent.runtime.state import AgentRuntimeState
    assert {"memory_save_results", "observations", "save_policy"} <= AgentRuntimeState.__annotations__.keys()

def test_memory_action_validates_content():
    from pydantic import ValidationError
    from src.web_app.agent.runtime.state import SupervisorAction
    with pytest.raises(ValidationError):
        SupervisorAction(action="memory_write", arguments={"content": ""})

def test_memory_result_distinguishes_failure():
    from src.web_app.agent.runtime.state import CapabilityResult
    result = CapabilityResult(action_id="save", capability="memory_write", status="failed", error="db_error")
    assert result.model_dump()["status"] == "failed"

def test_model_cannot_authorize_memory_save():
    from pydantic import ValidationError
    from src.web_app.agent.runtime.state import SupervisorAction
    with pytest.raises(ValidationError):
        SupervisorAction(action="memory_write", arguments={"content": "preference"}, save_policy={"write_memory": True})

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
