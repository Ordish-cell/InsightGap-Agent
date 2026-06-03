from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.mcp_repository import ToolCallRepository
from src.web_app.mcp.audit import tool_call_to_read
from src.web_app.mcp.registry import registry
from src.web_app.mcp.schemas import ToolCallRead
from src.web_app.mcp.tool_executor import tool_executor


class MCPService:
    def ensure_builtin_tools(self, db: Session) -> None:
        registry.ensure_builtin_tools(db)

    def list_tools(self, db: Session) -> list[dict[str, Any]]:
        return [item.model_dump() for item in registry.list_tools(db)]

    def get_tool(self, db: Session, tool_name: str) -> dict[str, Any] | None:
        tool = registry.get_tool(db, tool_name)
        return tool.model_dump() if tool else None

    def call_tool(self, db: Session, user_id: int, tool_name: str, input_data: dict[str, Any], agent_run_id: int | None = None, dry_run: bool = False) -> dict[str, Any]:
        return tool_executor.call_tool(db, user_id, tool_name, input_data, agent_run_id=agent_run_id, dry_run=dry_run).model_dump()

    def list_tool_calls(self, db: Session, user_id: int, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return [tool_call_to_read(item).model_dump() for item in ToolCallRepository(db).list_by_user(user_id, limit, offset)]

    def get_tool_call(self, db: Session, user_id: int, tool_call_id: int) -> dict[str, Any]:
        item = ToolCallRepository(db).get_by_user(user_id, tool_call_id)
        if not item:
            raise ValueError("ToolCall not found")
        return tool_call_to_read(item).model_dump()

    def health(self, db: Session | None = None) -> dict[str, Any]:
        return registry.health(db)


mcp_service = MCPService()
