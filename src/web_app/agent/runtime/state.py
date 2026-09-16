"""Serializable loop state; routing plans are deliberately not state channels."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReplyArguments(Arguments):
    instruction: str = ""


class QuestionArguments(Arguments):
    question: str = Field(min_length=1)


class ProposalArguments(QuestionArguments):
    query: str = Field(min_length=1)
    benefit: str = Field(min_length=1)


class SearchArguments(Arguments):
    query: str = Field(min_length=1)
    document_ids: list[int] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    overview: bool = False


class ResearchArguments(Arguments):
    query: str = Field(min_length=1)
    depth: Literal["quick", "standard", "deep"] = "standard"


class ToolArguments(Arguments):
    name: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)


class ArtifactArguments(Arguments):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    artifact_type: str = "markdown_report"


class MemoryArguments(Arguments):
    content: str = Field(min_length=1)
    memory_type: Literal["semantic", "episodic", "working"] = "semantic"
    importance: float = Field(default=0.8, ge=0, le=1)


class SkillArguments(Arguments):
    operation: Literal["match", "create_draft"]
    query: str = ""
    draft: dict[str, Any] = Field(default_factory=dict)


ARGUMENT_TYPES = {
    "respond": ReplyArguments,
    "ask_user": QuestionArguments,
    "propose_deep_research": ProposalArguments,
    "rag": SearchArguments,
    "deep_research": ResearchArguments,
    "tool": ToolArguments,
    "artifact": ArtifactArguments,
    "memory_search": SearchArguments,
    "memory_write": MemoryArguments,
    "skill": SkillArguments,
}


class SupervisorAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "respond",
        "ask_user",
        "propose_deep_research",
        "rag",
        "deep_research",
        "tool",
        "artifact",
        "memory_search",
        "memory_write",
        "skill",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)
    requires_quality_gate: bool = False

    @model_validator(mode="after")
    def validate_arguments(self):
        self.arguments = (
            ARGUMENT_TYPES[self.action].model_validate(self.arguments).model_dump()
        )
        return self



class CapabilityResult(BaseModel):
    action_id: str
    capability: str
    status: Literal[
        "ok", "empty", "degraded", "failed", "rejected", "blocked", "unknown"
    ]
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str = ""
    retryable: bool = False


class AgentRuntimeState(TypedDict, total=False):
    loop_protocol_version: int
    model_turn_id: str
    native_tool_call: dict | None
    native_preamble: str
    native_protocol_error: str
    native_final_text_id: str
    runtime_version: int
    user_id: int
    run_id: int
    thread_id: str
    conversation_id: str
    user_input: str
    mode: str
    source: str
    model_context: dict
    request: dict
    permission: dict
    status: str
    error: str
    errors: list
    context: dict
    page_context: dict
    conversation_files: list
    file_context: dict
    interaction_version: int
    messages: list
    current_action: dict
    observations: list
    runtime_budget: dict
    research_proposal: dict | None
    pending_research_proposal: dict | None
    research_authorized: bool
    research_confirmed: bool
    research_query: str
    save_policy: dict
    termination_reason: str
    finalization_started: bool
    route: str  # Public presentation only.
    risk_level: str  # Observed tool policy, never model-provided authority.
    research_result: dict
    rag_result: dict
    tool_result: dict
    tool_call: dict
    tool_calls: list
    artifacts: list
    memory_updates: list
    memory_save_results: list
    skill_drafts: list
    matched_skill: dict | None
    candidate_skills: list
    skill_reuse: dict
    approval_required: bool
    approval_payload: dict | None
    approval_pause_mode: str
    pending_approval_id: str | None
    pending_tool_call_id: int | None
    pending_tool_name: str | None
    pending_tool_args: dict | None
    resume_token: str | None
    resolved_tool_call_ids: list
    writes_denied: bool
    final_output: str
    final_answer: str
    final_payload: dict
    final_warnings: list
    evaluation: dict
    visible_thoughts: list
    langgraphstatus: dict
    pipeline_steps: list
    _answer_started_emitted: bool
    _answer_delta_emitted: bool
    _answer_completed_emitted: bool
