from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    user_input: str = ""
    input: str | None = None
    conversation_id: str | None = None
    mode: str = "react"
    run_type: str = "agent_runtime"
    route: Literal["research", "rag", "artifact", "skill", "memory", "tool"] | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    feed_card_id: int | None = None
    source: str = "agent_page"
    page_context: dict[str, Any] = Field(default_factory=dict)
    auto_skill: bool = True
    use_existing_skills: bool = True
    create_skill_draft_if_reusable: bool = True
    query: str | None = None
    depth: Literal["quick", "standard", "deep"] = "standard"
    save_artifact: bool = True
    write_memory: bool = True
    create_skill_draft: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    attachment_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
