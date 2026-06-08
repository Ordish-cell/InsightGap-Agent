from typing import Any

from sqlalchemy.orm import Session

from src.web_app.core.constants import L0_READ_ONLY, L1_DRAFT, L2_LOCAL_WRITE, L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.mcp_repository import MCPServerRepository, MCPToolRepository
from src.web_app.mcp.schemas import MCPToolRead, MCPToolSpec

BUILTIN_SERVER_NAME = "builtin_local_mcp"

BUILTIN_TOOLS = [
    MCPToolSpec(
        name="search_mcp.search",
        description="Deterministic local search fallback.",
        category="search",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"results": {"type": "array"}}},
    ),
    MCPToolSpec(
        name="github_mcp.repo_summary",
        description="Deterministic local GitHub repository summary.",
        category="github",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {"repo": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"repo": {"type": "string"}, "summary": {"type": "string"}}},
    ),
    MCPToolSpec(
        name="file_mcp.read_artifact",
        description="Read a user-owned local artifact.",
        category="file",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {"artifact_id": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"artifact_id": {"type": "integer"}, "content": {"type": "string"}}},
    ),
    MCPToolSpec(
        name="artifact_mcp.create_text_artifact",
        description="Create a local text artifact.",
        category="artifact",
        safety_level=L2_LOCAL_WRITE,
        input_schema={"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}, "artifact_type": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"artifact_id": {"type": "integer"}}},
    ),
    MCPToolSpec(
        name="memory_mcp.search",
        description="Search user-owned memory.",
        category="memory",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"memories": {"type": "array"}}},
    ),
    MCPToolSpec(
        name="memory_mcp.add",
        description="Add user-owned memory.",
        category="memory",
        safety_level=L2_LOCAL_WRITE,
        input_schema={"type": "object", "properties": {"content": {"type": "string"}, "memory_type": {"type": "string"}, "importance": {"type": "number"}}},
        output_schema={"type": "object", "properties": {"memory_id": {"type": "integer"}}},
    ),
    MCPToolSpec(
        name="skill_mcp.create_draft",
        description="Create a local skill draft.",
        category="skill",
        safety_level=L2_LOCAL_WRITE,
        input_schema={"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "trigger_text": {"type": "string"}, "tool_plan": {"type": "array"}}},
        output_schema={"type": "object", "properties": {"skill_id": {"type": "integer"}}},
    ),
    MCPToolSpec(
        name="email_mcp.create_draft",
        description="Create an email draft without sending.",
        category="email",
        safety_level=L1_DRAFT,
        input_schema={"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"draft": {"type": "object"}, "sent": {"type": "boolean"}}},
    ),
    MCPToolSpec(
        name="browser_mcp.plan_actions",
        description="Plan browser actions without execution.",
        category="browser",
        safety_level=L1_DRAFT,
        input_schema={"type": "object", "properties": {"goal": {"type": "string"}, "url": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"plan": {"type": "array"}, "executed": {"type": "boolean"}}},
    ),
    # ── Local file tools ──────────────────────────────────────
    MCPToolSpec(
        name="local_file.list",
        description="List files in the allowed workspace directory.",
        category="local_file",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}, "files": {"type": "array"}}},
    ),
    MCPToolSpec(
        name="local_file.read",
        description="Read a file in the allowed workspace directory.",
        category="local_file",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
    ),
    MCPToolSpec(
        name="local_file.write",
        description="Write a file in the allowed workspace directory. Requires approval (L3).",
        category="local_file",
        safety_level=L3_EXTERNAL_WRITE,
        requires_approval=True,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "string", "default": "create_or_overwrite"}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}, "written": {"type": "boolean"}}},
    ),
    MCPToolSpec(
        name="local_file.append",
        description="Append content to a file in the workspace. Requires approval (L3).",
        category="local_file",
        safety_level=L3_EXTERNAL_WRITE,
        requires_approval=True,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}, "written": {"type": "boolean"}}},
    ),
    MCPToolSpec(
        name="local_file.delete",
        description="Delete a file in the workspace. L4 – blocked by default.",
        category="local_file",
        safety_level=L4_HIGH_RISK,
        requires_approval=True,
        enabled=False,  # blocked by default
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}, "deleted": {"type": "boolean"}}},
    ),
    # ── Email send ────────────────────────────────────────────
    MCPToolSpec(
        name="email.send",
        description="Send an email. L3 – requires approval. Currently uses mock provider unless SMTP is configured.",
        category="email",
        safety_level=L3_EXTERNAL_WRITE,
        requires_approval=True,
        input_schema={"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}, "provider": {"type": "string"}, "message": {"type": "string"}}},
    ),
]


class MCPRegistry:
    def ensure_builtin_tools(self, db: Session) -> None:
        server_repo = MCPServerRepository(db)
        tool_repo = MCPToolRepository(db)
        server = server_repo.get_by_name(BUILTIN_SERVER_NAME)
        if not server:
            server = server_repo.create(name=BUILTIN_SERVER_NAME, description="Built-in deterministic local MCP provider", server_url="local://builtin", enabled=True, auth_config={"server_type": "builtin", "capabilities": ["local_tools"]})
        for spec in BUILTIN_TOOLS:
            existing = tool_repo.get_by_name(spec.name)
            values: dict[str, Any] = {
                "server_id": server.id,
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "permission_level": spec.safety_level,
                "approval_required": spec.requires_approval,
                "enabled": spec.enabled,
            }
            if existing:
                tool_repo.update(existing, **values)
            else:
                tool_repo.create(**values)

    def list_tools(self, db: Session) -> list[MCPToolRead]:
        self.ensure_builtin_tools(db)
        return [tool_to_read(tool) for tool in MCPToolRepository(db).list_enabled()]

    def get_tool(self, db: Session, tool_name: str) -> MCPToolRead | None:
        self.ensure_builtin_tools(db)
        tool = MCPToolRepository(db).get_by_name(tool_name)
        return tool_to_read(tool) if tool and tool.enabled else None

    def health(self, db: Session | None = None) -> dict[str, Any]:
        if not db:
            return {"status": "ok", "provider": "builtin_local_mcp", "external_servers": 0, "fallback_enabled": True}
        self.ensure_builtin_tools(db)
        return {"status": "ok", "provider": BUILTIN_SERVER_NAME, "tools_count": len(MCPToolRepository(db).list_enabled()), "external_servers": 0, "fallback_enabled": True}


def tool_to_read(tool) -> MCPToolRead:
    return MCPToolRead(
        id=tool.id,
        server_id=tool.server_id,
        name=tool.name,
        description=tool.description,
        category=(tool.name.split("_mcp.", 1)[0] if "_mcp." in tool.name else "local"),
        input_schema=tool.input_schema or {},
        output_schema=tool.output_schema or {},
        safety_level=tool.permission_level,
        enabled=tool.enabled,
        requires_approval=tool.approval_required,
        metadata={},
    )


registry = MCPRegistry()
