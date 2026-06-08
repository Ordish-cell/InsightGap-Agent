from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolActionPlan(BaseModel):
    """Structured plan for a tool action that may require approval."""
    intent: str = ""
    tool_name: str = ""
    risk_level: str = "L0"
    requires_approval: bool = False
    args: dict[str, Any] = Field(default_factory=dict)
    preview: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class MCPToolSpec(BaseModel):
    name: str
    description: str = ""
    category: str = "local"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    safety_level: str = "L0_READ_ONLY"
    enabled: bool = True
    requires_approval: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPToolRead(BaseModel):
    id: int | None = None
    server_id: int | None = None
    name: str
    description: str = ""
    category: str = "local"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    safety_level: str = "L0_READ_ONLY"
    enabled: bool = True
    requires_approval: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    agent_run_id: int | None = None
    dry_run: bool = False


class ToolCallRead(BaseModel):
    id: int
    user_id: int
    agent_run_id: int | None = None
    tool_id: int | None = None
    tool_name: str
    safety_level: str
    status: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    approval_id: int | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
