from typing import Any, Literal, TypedDict

AgentRoute = Literal["research", "rag", "artifact", "skill", "memory", "tool", "blocked", "approval"]


class AgentRuntimeState(TypedDict, total=False):
    user_id: int
    run_id: int
    user_input: str
    mode: str
    source: str
    page_context: dict[str, Any]
    route: AgentRoute
    status: str
    permission: dict[str, Any]
    context: dict[str, Any]
    rag: dict[str, Any]
    research: dict[str, Any]
    artifacts: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    skill_drafts: list[dict[str, Any]]
    matched_skill: dict[str, Any]
    candidate_skills: list[dict[str, Any]]
    created_skill_draft: dict[str, Any]
    skill_reuse: dict[str, Any]
    tool_call: dict[str, Any]
    evaluation: dict[str, Any]
    final_output: str
    error: str
    events: list[dict[str, Any]]
