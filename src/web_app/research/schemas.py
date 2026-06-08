from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str | None = None
    depth: Literal["quick", "standard", "deep"] = "standard"
    source: Literal["manual", "feed_card"] = "manual"
    feed_card_id: int | None = None
    card_snapshot: dict[str, Any] | None = None
    auto_start: bool = True
    save_artifact: bool = True
    write_memory: bool = True
    create_skill_draft: bool = True
    force_engine: Literal["open_deep_research", "fallback"] | None = None


class ResearchResult(BaseModel):
    summary: str
    findings: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    opportunities: list[dict[str, Any]] = Field(default_factory=list)
    suggested_actions: list[dict[str, Any]] = Field(default_factory=list)
    markdown_report: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchRunRead(BaseModel):
    id: str
    user_id: int
    feed_card_id: int | None = None
    agent_run_id: int | None = None
    query: str
    status: str
    summary: str = ""
    findings: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    opportunities: list[dict[str, Any]] = Field(default_factory=list)
    suggested_actions: list[dict[str, Any]] = Field(default_factory=list)
    markdown_report: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    artifact_id: int | None = None
    skill_draft_id: int | None = None
    error: str = ""
    error_message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    completed_at: str | None = None
