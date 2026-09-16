from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class LLMToolCall(BaseModel):
    """A single tool call selected by the LLM."""
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMToolSelectionResult(BaseModel):
    """Structured output from the LLM-based tool selector."""

    model_config = ConfigDict(extra="ignore")

    route: Literal["chat", "tool", "research", "rag", "artifact", "memory", "skill", "mixed", "unknown_tool"] = "chat"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    missing_fields: list[dict[str, str]] = Field(default_factory=list)
    # missing_fields items: {"tool_name": "...", "field": "...", "question": "..."}
    requested_action: str | None = None
    reason: str = ""
