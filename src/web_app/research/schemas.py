from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str | None = None
    depth: Literal["quick", "standard", "deep"] = "standard"
    save_artifact: bool = True
    write_memory: bool = True
    create_skill_draft: bool = True


class ResearchResult(BaseModel):
    summary: str
    findings: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    opportunities: list[dict[str, Any]] = Field(default_factory=list)
    suggested_actions: list[dict[str, Any]] = Field(default_factory=list)
    markdown_report: str
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
    artifact_id: int | None = None
    skill_draft_id: int | None = None
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    completed_at: str | None = None
