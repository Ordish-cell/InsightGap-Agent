from datetime import UTC, datetime
from typing import Any

from src.web_app.mcp.schemas import ToolCallRead


def tool_call_to_read(item, approval_id: int | None = None, completed_at: datetime | None = None) -> ToolCallRead:
    output = item.output or {}
    metadata = output.get("_metadata", {}) if isinstance(output, dict) else {}
    return ToolCallRead(
        id=item.id,
        user_id=item.user_id,
        agent_run_id=item.run_id,
        tool_id=item.mcp_tool_id,
        tool_name=item.tool_name,
        safety_level=item.permission_level,
        status=item.status,
        input=item.input or {},
        output={key: value for key, value in output.items() if key != "_metadata"} if isinstance(output, dict) else {},
        error=item.error_message or "",
        approval_id=approval_id or metadata.get("approval_id"),
        created_at=item.created_at,
        completed_at=completed_at or metadata.get("completed_at") or (datetime.now(UTC).replace(tzinfo=None) if item.status in {"completed", "failed", "blocked", "waiting_approval"} else None),
    )
