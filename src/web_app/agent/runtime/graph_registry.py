"""Runtime graph node registry.

This module keeps LangGraph node names in one place while preserving
RuntimeNodes as the only callable provider.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from src.web_app.agent.runtime.nodes import RuntimeNodes


SideEffectLevel = Literal["none", "read", "write", "external_tool", "final"]


@dataclass(frozen=True)
class RuntimeNodeSpec:
    name: str
    attr_name: str
    kind: str
    is_route_destination: bool = False
    runnable_enabled: bool = True
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    side_effect_level: SideEffectLevel = "none"
    description: str = ""


SETUP_NODE_NAMES: tuple[str, ...] = (
    "permission_guard",
    "home_intent_react",
    "planner",
)

READ_NODE_NAMES: tuple[str, ...] = (
    "parallel_prefetch",
    "parallel_read_stage",
    "supervisor_observer",
    # Registered for compatibility; context/skill execute inside parallel_read_stage.
    "context_builder",
    "skill_matcher",
)

AGENT_NODE_NAMES: tuple[str, ...] = (
    "research_agent",
    "rag_agent",
    "artifact_agent",
    "tool_agent",
    "memory_agent",
    "skill_agent",
)

EVAL_FINAL_NODE_NAMES: tuple[str, ...] = (
    "evaluator",
    "final_response",
)

GRAPH_NODE_NAMES: tuple[str, ...] = (
    *SETUP_NODE_NAMES,
    *READ_NODE_NAMES,
    *AGENT_NODE_NAMES,
    *EVAL_FINAL_NODE_NAMES,
)

ROUTE_DESTINATION_NODE_NAMES: tuple[str, ...] = (
    *AGENT_NODE_NAMES,
    "evaluator",
    "final_response",
)

RUNTIME_NODE_SPECS: tuple[RuntimeNodeSpec, ...] = (
    RuntimeNodeSpec(
        "permission_guard", "permission_guard", "setup",
        reads=("user_input",), writes=("permission",),
        description="Classify permission and safety level before planning.",
    ),
    RuntimeNodeSpec(
        "home_intent_react", "home_intent_react", "setup",
        reads=("user_input", "page_context"), writes=("home_intent",),
        side_effect_level="read",
        description="Infer lightweight home/page intent before route planning.",
    ),
    RuntimeNodeSpec(
        "planner", "planner", "setup",
        reads=("user_input", "home_intent", "page_context"),
        writes=("route_plan", "execution_plan", "route", "approval_required", "approval_payload"),
        side_effect_level="read",
        description="Create the route plan and execution plan for the run.",
    ),
    RuntimeNodeSpec(
        "parallel_prefetch", "parallel_prefetch", "read",
        reads=("user_input", "route_plan"), writes=("prefetch_results", "prefetch_warnings", "prefetch_elapsed_ms"),
        side_effect_level="read",
        description="Run low-risk read prefetch for RAG, memory, skill, and graph context.",
    ),
    RuntimeNodeSpec(
        "parallel_read_stage", "parallel_read_stage", "read",
        reads=("route_plan", "prefetch_results"),
        writes=("context", "matched_skill", "candidate_skills", "parallel_read_results", "parallel_read_warnings"),
        side_effect_level="read",
        description="Prepare context and skill data, with optional RAG evidence warmup.",
    ),
    RuntimeNodeSpec(
        "supervisor_observer", "supervisor_observer", "read",
        reads=("route_plan", "execution_plan", "completed_nodes", "parallel_read_results"),
        writes=("supervisor_decision", "supervisor_warnings", "supervisor_trace"),
        description="Observe runtime state for future supervisor control without changing dispatch.",
    ),
    RuntimeNodeSpec(
        "context_builder", "context_builder", "read",
        reads=("user_input", "prefetch_results"), writes=("context", "rag_evidence"),
        side_effect_level="read",
        description="Compatibility node for context building; main chain executes it inside parallel_read_stage.",
    ),
    RuntimeNodeSpec(
        "skill_matcher", "skill_matcher", "read",
        reads=("user_input", "context", "prefetch_results"), writes=("matched_skill", "candidate_skills"),
        side_effect_level="read",
        description="Compatibility node for skill matching; main chain executes it inside parallel_read_stage.",
    ),
    RuntimeNodeSpec(
        "research_agent", "research_agent", "agent", True,
        reads=("route_plan", "user_input"), writes=("research", "research_result", "agent_results"),
        side_effect_level="read",
        description="Run explicit research or lightweight research fallback.",
    ),
    RuntimeNodeSpec(
        "rag_agent", "rag_agent", "agent", True,
        reads=("user_input", "context", "parallel_read_results"), writes=("rag", "rag_result", "agent_results"),
        side_effect_level="read",
        description="Generate a RAG answer using prepared or freshly retrieved evidence.",
    ),
    RuntimeNodeSpec(
        "artifact_agent", "artifact_agent", "agent", True,
        reads=("route_plan", "final_output"), writes=("artifact_result", "artifacts", "agent_results"),
        side_effect_level="write",
        description="Create a file/artifact output for the run.",
    ),
    RuntimeNodeSpec(
        "tool_agent", "tool_agent", "agent", True,
        reads=("route_plan", "user_input"), writes=("tool_call", "tool_result", "approval_payload", "agent_results"),
        side_effect_level="external_tool",
        description="Execute or pause for approval around external tool calls.",
    ),
    RuntimeNodeSpec(
        "memory_agent", "memory_agent", "agent", True,
        reads=("user_input", "context"), writes=("memory_result", "memory_save_results", "memory_updates", "agent_results"),
        side_effect_level="write",
        description="Write explicit or inferred memory items.",
    ),
    RuntimeNodeSpec(
        "skill_agent", "skill_agent", "agent", True,
        reads=("user_input", "context"), writes=("skill_result", "skill_drafts", "created_skill_draft", "agent_results"),
        side_effect_level="write",
        description="Detect reusable skill patterns and create optional skill drafts.",
    ),
    RuntimeNodeSpec(
        "evaluator", "evaluator", "eval_final", True,
        reads=("agent_results", "rag_result", "tool_result", "memory_result"),
        writes=("evaluation_result", "final_response_constraints", "final_warnings"),
        description="Evaluate formal agent results and produce final response constraints.",
    ),
    RuntimeNodeSpec(
        "final_response", "final_response", "eval_final", True,
        reads=("final_output", "agent_results", "evaluation_result", "runtime_latency_trace"),
        writes=("final_answer", "final_payload", "status"),
        side_effect_level="final",
        description="Aggregate final payload and answer text for the user.",
    ),
)

FALLBACK_NODE_NAMES: tuple[str, ...] = (
    "permission_guard",
    "home_intent_react",
    "planner",
    "parallel_prefetch",
    "parallel_read_stage",
    "supervisor_observer",
    "research",
    "rag",
    "artifact",
    "skill_librarian",
    "tool",
    "memory_writer",
    "skill_draft_detector",
    "evaluator",
    "final_response",
)


def build_runtime_node_registry(nodes: RuntimeNodes) -> dict[str, Callable[..., Any]]:
    return {spec.name: getattr(nodes, spec.attr_name) for spec in RUNTIME_NODE_SPECS}


def build_fallback_nodes(nodes: RuntimeNodes) -> list[Callable[..., Any]]:
    return [getattr(nodes, name) for name in FALLBACK_NODE_NAMES]
