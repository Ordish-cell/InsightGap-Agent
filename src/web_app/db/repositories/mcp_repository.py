from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import MCPServer, MCPTool, ToolCall


class MCPServerRepository(BaseRepository[MCPServer]):
    model = MCPServer

    def get_by_name(self, name: str) -> MCPServer | None:
        return self.db.execute(select(MCPServer).where(MCPServer.name == name)).scalar_one_or_none()


class MCPToolRepository(BaseRepository[MCPTool]):
    model = MCPTool

    def get_by_name(self, name: str) -> MCPTool | None:
        return self.db.execute(select(MCPTool).where(MCPTool.name == name)).scalar_one_or_none()

    def list_enabled(self) -> list[MCPTool]:
        return list(self.db.execute(select(MCPTool).where(MCPTool.enabled.is_(True)).order_by(MCPTool.name)).scalars())


class ToolCallRepository(BaseRepository[ToolCall]):
    model = ToolCall

    def list_by_user(self, user_id: int, limit: int = 50, offset: int = 0) -> list[ToolCall]:
        stmt = select(ToolCall).where(ToolCall.user_id == user_id).order_by(ToolCall.created_at.desc()).limit(limit).offset(offset)
        return list(self.db.execute(stmt).scalars())

    def get_by_user(self, user_id: int, tool_call_id: int) -> ToolCall | None:
        return self.db.execute(select(ToolCall).where(ToolCall.user_id == user_id, ToolCall.id == tool_call_id)).scalar_one_or_none()

    def list_by_run(self, user_id: int, run_id: int) -> list[ToolCall]:
        stmt = select(ToolCall).where(ToolCall.user_id == user_id, ToolCall.run_id == run_id).order_by(ToolCall.created_at.desc())
        return list(self.db.execute(stmt).scalars())
