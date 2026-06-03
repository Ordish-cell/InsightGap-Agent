from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_id: int
    run_id: int
    user_input: str
    intent: str
    mode: str
    profile: dict[str, Any]
    context: str
    evidence: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    skill_drafts: list[dict[str, Any]]
    final_output: str
    error: str
