from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentTask(BaseModel):
    task_id: str
    agent: str
    purpose: str = ""
    input_requirements: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    can_parallel: bool = False
    required: bool = True
    max_retries: int = 1
    success_criteria: list[str] = Field(default_factory=list)
    fallback_agent: str | None = None


class ExecutionPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: f"plan:{uuid4().hex[:12]}")
    intent: str = "chat"
    answer_mode: str = "chat"
    risk_level: str = "L0"
    needs_approval: bool = False
    tasks: list[AgentTask] = Field(default_factory=list)
    final_synthesis: str = "answer"
    replan_allowed: bool = False
    max_replans: int = 0
    explicit_research: bool = False
    research_mode: str = "none"


class AgentResult(BaseModel):
    task_id: str = ""
    agent: str
    status: Literal["ok", "failed", "skipped", "needs_approval", "denied", "timeout"] = "ok"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    memory_updates: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    next_suggestions: list[str] = Field(default_factory=list)
    should_replan: bool = False
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateDelta(BaseModel):
    updates: dict[str, Any] = Field(default_factory=dict)
    append: dict[str, list[Any]] = Field(default_factory=dict)
    completed_node: str | None = None
    warnings: list[str] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    agent_result: AgentResult | dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeResult(BaseModel):
    node: str
    status: Literal["ok", "failed", "skipped", "needs_approval", "denied", "timeout"] = "ok"
    delta: StateDelta = Field(default_factory=StateDelta)
    summary: str = ""
    elapsed_ms: int | None = None


class EvaluationResult(BaseModel):
    pass_: bool = True
    score: float = Field(1.0, ge=0.0, le=1.0)
    missing: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_next_tasks: list[AgentTask] = Field(default_factory=list)
    final_response_constraints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    mode: Literal["observe_only"] = "observe_only"
    current_intent: str | None = None
    current_route: list[str] = Field(default_factory=list)
    next_expected_node: str | None = None
    observed_completed_nodes: list[str] = Field(default_factory=list)
    observed_pending_nodes: list[str] = Field(default_factory=list)
    has_prefetch_context: bool = False
    has_parallel_read_context: bool = False
    has_rag_prepare_evidence: bool = False
    rag_prepare_evidence_count: int = 0
    failed_agents: list[str] = Field(default_factory=list)
    waiting_approval: bool = False
    should_replan_hint: bool = False
    replan_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


def execution_plan_from_route_plan(route_plan: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap the legacy route_plan in a minimal ExecutionPlan-compatible dict."""
    state = state or {}
    route = list(route_plan.get("route") or [])
    tasks = [
        AgentTask(
            task_id=f"{index}:{agent}",
            agent=str(agent),
            purpose=_purpose_for_agent(str(agent), route_plan),
            success_criteria=_success_criteria_for_agent(str(agent)),
        )
        for index, agent in enumerate(route)
    ]
    plan = ExecutionPlan(
        intent=str(route_plan.get("intent") or "chat"),
        answer_mode=str(route_plan.get("answer_mode") or "chat"),
        risk_level=str(route_plan.get("risk_level") or "L0"),
        needs_approval=bool(route_plan.get("needs_approval", False)),
        tasks=tasks,
        final_synthesis=str(route_plan.get("expected_output") or "answer"),
        explicit_research=bool(route_plan.get("explicit_research", False)),
        research_mode=str(route_plan.get("research_mode") or "none"),
    )
    return plan.model_dump()


def append_agent_result(state: dict[str, Any], result: AgentResult | dict[str, Any]) -> dict[str, Any]:
    payload = result.model_dump() if isinstance(result, AgentResult) else AgentResult(**result).model_dump()
    existing = state.setdefault("agent_results", [])
    for item in existing:
        if not isinstance(item, dict):
            continue
        if (
            item.get("agent") == payload.get("agent")
            and item.get("status") == payload.get("status")
            and item.get("task_id") == payload.get("task_id")
            and item.get("summary") == payload.get("summary")
        ):
            return state
    existing.append(payload)
    return state


def task_id_for_agent(state: dict[str, Any], agent: str) -> str:
    plan = state.get("execution_plan") or {}
    for task in plan.get("tasks", []) or []:
        if task.get("agent") == agent:
            return str(task.get("task_id") or "")
    return ""


def _purpose_for_agent(agent: str, route_plan: dict[str, Any]) -> str:
    intent = str(route_plan.get("intent") or "chat")
    return {
        "research_agent": "Run explicit research task" if route_plan.get("explicit_research") else "Run lightweight research fallback",
        "rag_agent": "Retrieve knowledge-base evidence for the user request",
        "artifact_agent": "Create requested artifact",
        "tool_agent": "Execute or prepare tool action",
        "memory_agent": "Write or extract useful memory",
        "skill_agent": "Detect reusable workflow",
        "evaluator": "Evaluate runtime outputs and constraints",
        "final_response": "Synthesize final user-facing answer",
    }.get(agent, f"Execute {agent} for {intent}")


def _success_criteria_for_agent(agent: str) -> list[str]:
    return {
        "rag_agent": ["answer or evidence returned"],
        "memory_agent": ["memory write status recorded"],
        "tool_agent": ["tool action completed or approval requested"],
        "artifact_agent": ["artifact result recorded"],
        "evaluator": ["constraints and warnings recorded"],
        "final_response": ["final answer produced"],
    }.get(agent, [])
