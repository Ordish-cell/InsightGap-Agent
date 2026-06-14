import logging
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.core.constants import L0_READ_ONLY, L1_DRAFT, L2_LOCAL_WRITE, L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.mcp_repository import MCPServerRepository, MCPToolRepository
from src.web_app.mcp.schemas import MCPToolRead, MCPToolSpec

logger = logging.getLogger(__name__)

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
        name="web.search",
        description="Search the public web for current information. Read-only lightweight search, separate from Open Deep Research.",
        category="web",
        safety_level=L0_READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "recency_days": {"type": "integer"},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "provider": {"type": "string"},
                "used_fallback": {"type": "boolean"},
                "final_query": {"type": "string"},
                "search_rounds": {"type": "array"},
                "reasoning_summary": {"type": "string"},
                "results": {"type": "array"},
                "error": {"type": "string"},
            },
        },
    ),
    MCPToolSpec(
        name="system.time",
        description="Read the local system date, time, weekday, timezone, and ISO timestamp. No web access.",
        category="system",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "time": {"type": "string"},
                "weekday": {"type": "string"},
                "timezone": {"type": "string"},
                "iso": {"type": "string"},
            },
        },
    ),
    MCPToolSpec(
        name="system.calc",
        description="Evaluate a simple local arithmetic expression. No web access.",
        category="system",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        output_schema={"type": "object", "properties": {"expression": {"type": "string"}, "result": {"type": "number"}}},
    ),
    MCPToolSpec(
        name="system.unit_convert",
        description="Convert common local units. No web access.",
        category="system",
        safety_level=L0_READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from": {"type": "string"},
                "to": {"type": "string"},
            },
            "required": ["value", "from", "to"],
        },
        output_schema={"type": "object", "properties": {"value": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}, "result": {"type": "number"}}},
    ),
    MCPToolSpec(
        name="system.uuid",
        description="Generate a UUID locally. No web access.",
        category="system",
        safety_level=L0_READ_ONLY,
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {"uuid": {"type": "string"}}},
    ),
    MCPToolSpec(
        name="system.hash",
        description="Hash text locally with md5, sha1, sha256, or sha512. No web access.",
        category="system",
        safety_level=L0_READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "algorithm": {"type": "string", "default": "sha256"},
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
        output_schema={"type": "object", "properties": {"algorithm": {"type": "string"}, "hash": {"type": "string"}}},
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
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "string", "default": "create_or_overwrite"}}, "required": ["path", "content"]},
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
        input_schema={
            "type": "object",
            "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "cc": {"type": "string"}, "bcc": {"type": "string"}},
            "required": ["to", "subject", "body"],
        },
        output_schema={"type": "object", "properties": {"success": {"type": "boolean"}, "sent": {"type": "boolean"}, "provider": {"type": "string"}, "message": {"type": "string"}, "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "body_preview": {"type": "string"}}},
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


# ── Alias → canonical name normalization ──────────────────────────
_ALIAS_TO_CANONICAL: dict[str, str] | None = None


def _collect_all_aliases() -> dict[str, list[str]]:
    """Collect aliases from provider modules."""
    from src.web_app.mcp.email_provider import EMAIL_TOOL_ALIASES
    from src.web_app.mcp.local_file_tools import LOCAL_FILE_TOOL_ALIASES
    from src.web_app.mcp.web_search_provider import WEB_SEARCH_TOOL_ALIASES
    merged: dict[str, list[str]] = {}
    merged.update(EMAIL_TOOL_ALIASES)
    merged.update(LOCAL_FILE_TOOL_ALIASES)
    merged.update(WEB_SEARCH_TOOL_ALIASES)
    merged.update({
        "system.time": ["runtime.now", "clock.now", "time.now", "current_time"],
        "system.calc": ["calculator", "calc"],
        "system.unit_convert": ["unit.convert", "unit_convert"],
        "system.uuid": ["uuid", "uuid.generate"],
        "system.hash": ["hash", "hash.text"],
    })
    return merged


def _build_alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    all_aliases = _collect_all_aliases()
    for canonical, aliases in all_aliases.items():
        mapping[canonical.lower()] = canonical
        for alias in aliases:
            key = alias.strip().lower()
            if key and key not in mapping:
                mapping[key] = canonical
    return mapping


def normalize_tool_name(name: str) -> str:
    """Map alias / short name to canonical tool name. Returns original if no match."""
    if not name:
        return name
    global _ALIAS_TO_CANONICAL
    if _ALIAS_TO_CANONICAL is None:
        _ALIAS_TO_CANONICAL = _build_alias_map()
    key = name.strip().lower()
    resolved = _ALIAS_TO_CANONICAL.get(key, name)
    if resolved != name:
        logger.debug("Tool name normalized: %r → %r", name, resolved)
    return resolved


registry = MCPRegistry()
