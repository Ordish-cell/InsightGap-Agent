from typing import Any, Literal, TypedDict

AgentRoute = Literal["research", "rag", "artifact", "skill", "memory", "tool", "blocked", "approval", "chat", "feed_research", "mixed"]

AgentIntent = Literal["chat", "research", "rag", "artifact", "tool", "memory", "skill", "feed_research", "mixed"]

RiskLevel = Literal["L0", "L1", "L2", "L3", "L4"]


class RoutePlan(TypedDict, total=False):
    intent: AgentIntent
    route: list[str]
    risk_level: RiskLevel
    needs_approval: bool
    expected_output: str
    reason: str


class AgentRuntimeState(TypedDict, total=False):
    # ── Identity ───────────────────────────────────────────────────
    user_id: int
    run_id: int
    thread_id: str
    conversation_id: str | None
    session_id: str | None
    user_input: str
    query: str
    mode: str
    source: str

    # ── Planner output ─────────────────────────────────────────────
    route_plan: RoutePlan
    home_intent: dict[str, Any]
    current_node: str
    completed_nodes: list[str]
    errors: list[dict[str, Any]]

    # ── Permission ─────────────────────────────────────────────────
    route: AgentRoute  # legacy — kept for backward compat
    status: str
    permission: dict[str, Any]
    approval_required: bool
    approval_payload: dict[str, Any] | None

    # ── Context ────────────────────────────────────────────────────
    page_context: dict[str, Any]
    context: dict[str, Any]
    context_packets: list[dict[str, Any]]
    selected_memories: list[dict[str, Any]]
    matched_skills: list[dict[str, Any]]

    # ── Agent outputs (structured) ─────────────────────────────────
    research_result: dict[str, Any] | None
    rag_result: dict[str, Any] | None
    artifact_result: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    memory_result: dict[str, Any] | None
    skill_result: dict[str, Any] | None
    evaluation_result: dict[str, Any] | None

    # ── Agent outputs (list form) ──────────────────────────────────
    agent_outputs: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    skill_drafts: list[dict[str, Any]]

    # ── Skill / legacy compat ──────────────────────────────────────
    rag: dict[str, Any]
    research: dict[str, Any]
    matched_skill: dict[str, Any] | None
    candidate_skills: list[dict[str, Any]]
    created_skill_draft: dict[str, Any] | None
    skill_reuse: dict[str, Any]
    tool_call: dict[str, Any]
    evaluation: dict[str, Any]

    # ── Final output ───────────────────────────────────────────────
    final_output: str
    final_answer: str | None
    final_payload: dict[str, Any] | None
    error: str
    events: list[dict[str, Any]]


# ── Helper functions for state manipulation ──────────────────────────

def append_output(state: AgentRuntimeState, node: str, output: dict[str, Any]) -> AgentRuntimeState:
    """Append a structured output from a node to agent_outputs."""
    entry = {"node": node, **output}
    state.setdefault("agent_outputs", []).append(entry)
    return state


def append_error(state: AgentRuntimeState, node: str, error_msg: str) -> AgentRuntimeState:
    """Record a non-fatal error from a node."""
    state.setdefault("errors", []).append({"node": node, "error": error_msg})
    return state


def mark_completed(state: AgentRuntimeState, node: str) -> AgentRuntimeState:
    """Mark a node as completed."""
    state.setdefault("completed_nodes", []).append(node)
    state["current_node"] = node
    return state
