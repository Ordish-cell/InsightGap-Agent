from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.mcp_repository import MCPToolRepository, ToolCallRepository
from src.web_app.mcp.audit import tool_call_to_read
from src.web_app.mcp.local_provider import local_provider
from src.web_app.mcp.registry import registry
from src.web_app.mcp.schemas import ToolCallRead
from src.web_app.services.permission_service import PermissionGuard


class MCPToolExecutor:
    def call_tool(self, db: Session, user_id: int, tool_name: str, input_data: dict[str, Any], agent_run_id: int | None = None, dry_run: bool = False) -> ToolCallRead:
        registry.ensure_builtin_tools(db)
        tool = MCPToolRepository(db).get_by_name(tool_name)
        safety_level = tool.permission_level if tool else L4_HIGH_RISK
        call = ToolCallRepository(db).create(user_id=user_id, run_id=agent_run_id, tool_name=tool_name, mcp_tool_id=tool.id if tool else None, input=input_data, output={}, permission_level=safety_level, status="pending", error_message="")

        if not tool or not tool.enabled:
            return self._finish(db, call, "failed", {}, "Tool not found")

        decision = PermissionGuard().check_tool_call(tool_name, tool.permission_level, tool.approval_required)
        if tool.permission_level == L4_HIGH_RISK or not decision["allowed"] and not decision["requires_approval"]:
            return self._finish(db, call, "blocked", {}, decision["reason"])
        if tool.permission_level == L3_EXTERNAL_WRITE or decision["requires_approval"]:
            approval = ApprovalRepository(db).create(user_id=user_id, run_id=agent_run_id, approval_type="mcp_tool_call", title=f"MCP tool approval required: {tool_name}", description=f"Tool {tool_name} requires approval before execution.", payload={"tool_call_id": call.id, "tool_name": tool_name, "input": input_data, "safety_level": tool.permission_level})
            return self._finish(db, call, "waiting_approval", {"_metadata": {"approval_id": approval.id}}, "approval_required", approval_id=approval.id)
        if dry_run:
            return self._finish(db, call, "completed", {"dry_run": True, "would_call": tool_name}, "")

        ToolCallRepository(db).update(call, status="running")
        try:
            output = local_provider.call(db, user_id, tool_name, input_data, agent_run_id)
            return self._finish(db, call, "completed", output, "")
        except Exception as exc:
            return self._finish(db, call, "failed", {}, str(exc))

    def _finish(self, db: Session, call, status: str, output: dict[str, Any], error: str, approval_id: int | None = None) -> ToolCallRead:
        completed_at = datetime.now(UTC).replace(tzinfo=None)
        output = dict(output or {})
        output.setdefault("_metadata", {})
        output["_metadata"]["completed_at"] = completed_at.isoformat()
        if approval_id:
            output["_metadata"]["approval_id"] = approval_id
        updated = ToolCallRepository(db).update(call, status=status, output=output, error_message=error)
        return tool_call_to_read(updated, approval_id=approval_id, completed_at=completed_at)


tool_executor = MCPToolExecutor()
