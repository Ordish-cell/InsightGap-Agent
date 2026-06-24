from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.artifact_repository import ArtifactRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.mcp_repository import MCPToolRepository, ToolCallRepository
from src.web_app.models.orm import User
from src.web_app.services import agent_service
from src.web_app.services.approval_service import update_approval_status
from src.web_app.services.artifact_service import artifact_service
from src.web_app.services.mcp_service import mcp_service
from src.web_app.tests.db_test_utils import make_test_session


def _user(db, email: str = "mcp-stage7@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_mcp_registry_seeds_builtin_tools():
    db = make_test_session()

    tools = mcp_service.list_tools(db)
    names = {tool["name"] for tool in tools}

    assert "search_mcp.search" in names
    assert "artifact_mcp.create_text_artifact" in names
    assert "browser_mcp.plan_actions" in names
    assert all(tool["enabled"] for tool in tools)


def test_mcp_search_tool_records_tool_call():
    db = make_test_session()
    user = _user(db)

    result = mcp_service.call_tool(db, user.id, "search_mcp.search", {"query": "langgraph", "limit": 2})

    assert result["status"] == "completed"
    assert result["output"]["results"][0]["source_type"] == "local_stub"
    calls = ToolCallRepository(db).list_by_user(user.id)
    assert calls[0].tool_name == "search_mcp.search"
    assert calls[0].status == "completed"


def test_mcp_artifact_read_is_user_scoped():
    db = make_test_session()
    owner = _user(db, "artifact-owner@example.com")
    other = _user(db, "artifact-other@example.com")
    path = artifact_service.save_text_artifact(owner.id, "owned_artifact.md", "owner only")
    artifact = ArtifactRepository(db).create(user_id=owner.id, artifact_type="note", title="Owned", file_path=path, metadata_json={})

    denied = mcp_service.call_tool(db, other.id, "file_mcp.read_artifact", {"artifact_id": artifact.id})
    allowed = mcp_service.call_tool(db, owner.id, "file_mcp.read_artifact", {"artifact_id": artifact.id})

    assert denied["status"] == "failed"
    assert "Artifact not found" in denied["error"]
    assert allowed["status"] == "completed"
    assert allowed["output"]["content"] == "owner only"


def test_artifact_detail_includes_owned_text_content():
    db = make_test_session()
    user = _user(db, "artifact-detail@example.com")
    path = artifact_service.save_text_artifact(user.id, "detail_artifact.md", "visible content")
    artifact = ArtifactRepository(db).create(user_id=user.id, artifact_type="note", title="Detail", file_path=path, metadata_json={})

    data = artifact_service.get_artifact(artifact.id, user.id, db)

    assert data["content"] == "visible content"


def test_mcp_local_write_tools_create_owned_records():
    db = make_test_session()
    user = _user(db)

    artifact = mcp_service.call_tool(db, user.id, "artifact_mcp.create_text_artifact", {"title": "MCP Note", "content": "hello", "artifact_type": "note"})
    memory = mcp_service.call_tool(db, user.id, "memory_mcp.add", {"content": "remember mcp", "importance": 0.8})
    skill = mcp_service.call_tool(db, user.id, "skill_mcp.create_draft", {"name": "MCP Skill", "description": "draft", "trigger_text": "mcp", "tool_plan": []})

    assert artifact["output"]["artifact_id"]
    assert memory["output"]["memory_id"]
    assert skill["output"]["skill_id"]


def test_mcp_l3_requires_approval_and_l4_is_blocked():
    db = make_test_session()
    user = _user(db)
    mcp_service.ensure_builtin_tools(db)
    repo = MCPToolRepository(db)
    repo.create(name="email_mcp.send_real", description="blocked external send", input_schema={}, output_schema={}, permission_level=L3_EXTERNAL_WRITE, approval_required=True, enabled=True)
    repo.create(name="system_mcp.delete_all", description="high risk", input_schema={}, output_schema={}, permission_level=L4_HIGH_RISK, approval_required=False, enabled=True)

    approval = mcp_service.call_tool(db, user.id, "email_mcp.send_real", {"to": "x@example.com"})
    blocked = mcp_service.call_tool(db, user.id, "system_mcp.delete_all", {})

    assert approval["status"] == "waiting_approval"
    assert approval["approval_id"]
    assert blocked["status"] == "blocked"
    assert blocked["error"] == "high_risk_denied"


def test_tool_args_hash_is_canonical():
    from src.web_app.mcp.tool_executor import hash_tool_args

    left = {"b": 2, "a": {"z": 1, "y": [3, 2]}}
    right = {"a": {"y": [3, 2], "z": 1}, "b": 2}

    assert hash_tool_args(left) == hash_tool_args(right)


def test_l3_prepare_is_idempotent_with_key():
    db = make_test_session()
    user = _user(db, "mcp-idempotent@example.com")
    payload = {"to": "x@example.com", "subject": "Hi", "body": "Body"}

    first = mcp_service.call_tool(db, user.id, "email.send", payload, idempotency_key="idem-email-1")
    second = mcp_service.call_tool(db, user.id, "email.send", payload, idempotency_key="idem-email-1")

    assert first["status"] == "waiting_approval"
    assert second["status"] == "waiting_approval"
    assert second["id"] == first["id"]
    assert second["approval_id"] == first["approval_id"]
    assert len(ToolCallRepository(db).list_by_user(user.id)) == 1
    assert len(ApprovalRepository(db).list_by_user(user.id)) == 1


def test_standalone_l3_approve_executes_tool_once():
    db = make_test_session()
    user = _user(db, "mcp-approve-once@example.com")
    payload = {"to": "x@example.com", "subject": "Hi", "body": "Body"}

    prepared = mcp_service.call_tool(db, user.id, "email.send", payload, idempotency_key="idem-email-approve")
    approved = update_approval_status(db, user.id, prepared["approval_id"], "approved")

    assert approved["status"] == "approved"
    assert approved["tool_result"]["success"] is True
    call = ToolCallRepository(db).get_by_user(user.id, prepared["id"])
    assert call.status == "completed"

    from src.web_app.mcp.tool_executor import tool_executor

    second = tool_executor.execute_approved_tool_once(db, user.id, prepared["id"], "email.send", payload)
    assert second["success"] is True
    assert len(ToolCallRepository(db).list_by_user(user.id)) == 1


def test_email_and_browser_tools_are_drafts_only():
    db = make_test_session()
    user = _user(db)

    email = mcp_service.call_tool(db, user.id, "email_mcp.create_draft", {"to": "x@example.com", "subject": "Hi", "body": "Draft"})
    browser = mcp_service.call_tool(db, user.id, "browser_mcp.plan_actions", {"goal": "inspect docs", "url": "https://example.com"})

    assert email["status"] == "completed"
    assert email["output"]["sent"] is False
    assert browser["output"]["executed"] is False


def test_agent_runtime_tool_node_records_tool_call():
    db = make_test_session()
    user = _user(db)

    result = agent_service.run_agent(db, user.id, {"user_input": "run local search", "route": "tool", "tool_name": "search_mcp.search", "tool_input": {"query": "agent os", "limit": 1}})

    assert result["status"] == "completed"
    assert result["route"] == "tool"
    assert result["tool_call"]["tool_name"] == "search_mcp.search"
    assert ToolCallRepository(db).list_by_run(user.id, result["run_id"])


def test_mcp_health_reports_builtin_provider():
    db = make_test_session()

    health = mcp_service.health(db)

    assert health["status"] == "ok"
    assert health["provider"] == "builtin_local_mcp"
    assert health["tools_count"] >= 9
