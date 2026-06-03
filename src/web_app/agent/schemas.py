from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    user_input: str
    mode: str = "react"
    run_type: str = "agent_runtime"
    route: Literal["research", "rag", "artifact", "skill", "memory", "tool"] | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    feed_card_id: int | None = None
    query: str | None = None
    depth: Literal["quick", "standard", "deep"] = "standard"
    save_artifact: bool = True
    write_memory: bool = True
    create_skill_draft: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    metadata: dict[str, Any] = Field(default_factory=dict)
