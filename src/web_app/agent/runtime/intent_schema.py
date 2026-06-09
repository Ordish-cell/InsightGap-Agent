import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

IntentName = Literal[
    "chat",
    "research",
    "rag",
    "artifact",
    "feed_research",
    "tool",
    "tool.email",
    "tool.local_file",
    "tool.browser",
    "tool.comment",
    "tool.form_submit",
    "tool.shell_readonly",
    "tool.shell_write",
    "tool.dangerous",
    "memory",
    "skill",
    "mixed",
]
RiskName = Literal["L0", "L1", "L2", "L3", "L4"]

ALLOWED_AGENTS = {
    "context_builder",
    "skill_matcher",
    "research_agent",
    "rag_agent",
    "artifact_agent",
    "tool_agent",
    "memory_agent",
    "skill_agent",
    "evaluator",
    "final_response",
}

# ── Agent alias normalization ──────────────────────────────────────
# LLM intent results may return short names (research, rag, artifact)
# but graph nodes are registered with _agent suffix.
AGENT_ALIAS_MAP: dict[str, str] = {
    "research": "research_agent",
    "deep_research": "research_agent",
    "rag": "rag_agent",
    "retrieval": "rag_agent",
    "document": "rag_agent",
    "artifact": "artifact_agent",
    "artifacts": "artifact_agent",
    "tool": "tool_agent",
    "mcp": "tool_agent",
    "memory": "memory_agent",
    "skill": "skill_agent",
    "chat": "final_response",
    "context": "context_builder",
    "skill_match": "skill_matcher",
    "eval": "evaluator",
    "evaluate": "evaluator",
}


def normalize_agent_name(name: str) -> str:
    """Map short / alias agent names to registered graph node names."""
    if not name:
        return ""
    key = name.strip().lower()
    return AGENT_ALIAS_MAP.get(key, key)


logger = logging.getLogger(__name__)


class HomeIntentResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: IntentName = "chat"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    risk_level: RiskName = "L0"
    needs_approval: bool = False
    needs_clarification: bool = False
    required_agents: list[str] = Field(default_factory=list)
    expected_output: str = "answer"
    reason_summary: str = ""
    suggested_route_hints: list[str] = Field(default_factory=list)
    tool_action_type: str | None = None
    model_used: str | None = None
    fallback_used: bool = False
    raw_intent_source: Literal["llm", "rule", "fallback"] = "rule"

    @field_validator("required_agents", "suggested_route_hints", mode="before")
    @classmethod
    def filter_unknown_agents(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        filtered: list[str] = []
        for raw in value:
            if not isinstance(raw, str):
                continue
            normalized = normalize_agent_name(raw)
            if normalized in ALLOWED_AGENTS and normalized not in filtered:
                filtered.append(normalized)
            else:
                logger.warning(
                    "Unknown agent filtered from intent result",
                    extra={"raw_agent": raw, "normalized_agent": normalized},
                )
        return filtered

    @field_validator("reason_summary")
    @classmethod
    def trim_reason(cls, value: str) -> str:
        return (value or "")[:240]

    def to_home_intent_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["detected_intent"] = data["intent"]
        data["required_capabilities"] = [agent.replace("_agent", "") for agent in self.required_agents if agent.endswith("_agent")]
        data["reasoning_summary"] = data["reason_summary"]
        return data


# ── LLM tool selection output schemas ────────────────────────────

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
