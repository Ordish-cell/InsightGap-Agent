"""LLM-owned route_plan supervision for the existing Agent Runtime.

The dispatcher still reads only route_plan. This module can replace that
route_plan in full mode after deterministic guardrails validate the LLM output.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.web_app.agent.llm.factory import get_chat_model, get_chat_model_by_name
from src.web_app.agent.runtime.llm_supervisor_prompts import (
    WEB_APP_LLM_SUPERVISOR_SYSTEM_PROMPT,
)
from src.web_app.agent.runtime.schemas import execution_plan_from_route_plan
from src.web_app.agent.runtime.state import AgentRuntimeState
from src.web_app.core.config import get_settings


SupervisorMode = Literal["off", "shadow", "full"]
TargetRuntime = Literal["web_app", "odr"]
RiskLevel = Literal["L0", "L1", "L2", "L3"]

SIDE_EFFECT_NODES = {
    "tool_agent",
    "artifact_agent",
    "memory_agent",
    "skill_agent",
    "research_agent",
}
DEEP_RESEARCH_TERMS = (
    "deep_research",
    "deep research",
    "深度研究",
    "研究报告",
    "long-form research",
    "comprehensive research",
    "open_deep_research",
    "odr",
)
EXPLICIT_ACTION_FIELDS = (
    "explicit_route",
    "explicit_agent",
    "selected_agent",
    "selected_action",
    "user_clicked_action",
    "action_id",
    "target_agent",
    "workflow",
)
ACTION_TO_NODE = {
    "artifact": "artifact_agent",
    "artifact_agent": "artifact_agent",
    "create_artifact": "artifact_agent",
    "tool": "tool_agent",
    "tool_agent": "tool_agent",
    "memory": "memory_agent",
    "memory_agent": "memory_agent",
    "skill": "skill_agent",
    "skill_agent": "skill_agent",
    "rag": "rag_agent",
    "document": "rag_agent",
    "document_qa": "rag_agent",
    "research": "research_agent",
    "research_agent": "research_agent",
    "deep_research": "research_agent",
    "chat": "final_response",
    "assistant": "final_response",
    "assistant_agent": "final_response",
    "final": "final_response",
    "final_response": "final_response",
}
NODE_ALIASES = {
    "assistant_agent": "final_response",
    "assistant": "final_response",
    "document_agent": "rag_agent",
    "approval_required": "final_response",
    "odr_deep_research": "research_agent",
    "open_deep_research": "research_agent",
}


class LLMSupervisorRouteDecision(BaseModel):
    target_runtime: TargetRuntime = "web_app"
    route: list[str] = Field(default_factory=list)
    next_node: str | None = None
    agent_name: str | None = None
    reason: str = ""
    risk_level: RiskLevel = "L0"
    requires_approval: bool = False
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedSupervisorRouteDecision(BaseModel):
    target_runtime: TargetRuntime
    route: list[str]
    next_node: str | None
    agent_name: str | None
    reason: str
    risk_level: RiskLevel
    requires_approval: bool
    confidence: float | None = None
    explicit_override: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    raw_decision: dict[str, Any] | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    original_planner_route: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMSupervisorSettings(BaseModel):
    enabled: bool
    mode: SupervisorMode
    model: str
    temperature: float
    timeout_seconds: int


async def llm_supervisor_route_node(
    state: AgentRuntimeState,
    config: dict[str, Any] | None = None,
) -> AgentRuntimeState:
    settings = resolve_llm_supervisor_settings(config)
    state.setdefault("llm_supervisor_trace", [])

    if not settings.enabled or settings.mode == "off":
        _append_trace(
            state,
            event="llm_supervisor_skipped",
            mode=settings.mode,
            reason="disabled" if not settings.enabled else "off",
        )
        return state

    planner_route = _planner_route(state)
    available_nodes = get_available_runtime_nodes(state)

    pre_rule = _build_pre_rule_decision(state, available_nodes, planner_route)
    if pre_rule is not None:
        return apply_supervisor_route_plan(state, pre_rule, mode=settings.mode)

    raw_output: dict[str, Any] | None = None
    try:
        decision = await _invoke_llm_supervisor(
            state,
            settings=settings,
            available_nodes=available_nodes,
            planner_route=planner_route,
        )
        raw_output = decision.model_dump()
        normalized = validate_and_normalize_llm_supervisor_route(
            decision,
            state,
            available_nodes=available_nodes,
            planner_route=planner_route,
        )
    except Exception as exc:
        normalized = build_fallback_supervisor_decision(
            state,
            reason=f"llm_supervisor_failed:{type(exc).__name__}:{exc}",
            planner_route=planner_route,
            available_nodes=available_nodes,
        )

    if raw_output is not None:
        normalized.raw_decision = raw_output
    normalized.metadata.setdefault("model", settings.model or "default:planner")
    normalized.metadata.setdefault("mode", settings.mode)
    return apply_supervisor_route_plan(state, normalized, mode=settings.mode)


def resolve_llm_supervisor_settings(config: dict[str, Any] | None = None) -> LLMSupervisorSettings:
    app_settings = get_settings()
    configurable = {}
    if isinstance(config, dict):
        maybe_configurable = config.get("configurable")
        if isinstance(maybe_configurable, dict):
            configurable = maybe_configurable
        else:
            configurable = config

    enabled = _bool_config(
        configurable.get("agent_llm_supervisor_enabled"),
        getattr(app_settings, "agent_llm_supervisor_enabled", False),
    )
    mode = str(
        configurable.get("agent_llm_supervisor_mode")
        or getattr(app_settings, "agent_llm_supervisor_mode", "shadow")
        or "shadow"
    ).lower()
    if mode not in {"off", "shadow", "full"}:
        mode = "shadow"
    return LLMSupervisorSettings(
        enabled=enabled,
        mode=mode,  # type: ignore[arg-type]
        model=str(
            configurable.get("agent_llm_supervisor_model")
            or getattr(app_settings, "agent_llm_supervisor_model", "")
            or ""
        ),
        temperature=float(
            configurable.get("agent_llm_supervisor_temperature")
            if configurable.get("agent_llm_supervisor_temperature") is not None
            else getattr(app_settings, "agent_llm_supervisor_temperature", 0)
        ),
        timeout_seconds=int(
            configurable.get("agent_llm_supervisor_timeout_seconds")
            if configurable.get("agent_llm_supervisor_timeout_seconds") is not None
            else getattr(app_settings, "agent_llm_supervisor_timeout_seconds", 20)
        ),
    )


def get_available_runtime_nodes(state: AgentRuntimeState | None = None) -> set[str]:
    from src.web_app.agent.runtime.graph_registry import ROUTE_DESTINATION_NODE_NAMES

    nodes = set(ROUTE_DESTINATION_NODE_NAMES)
    if state and isinstance(state.get("available_nodes"), list):
        nodes.update(str(item) for item in state["available_nodes"] if item)
    return nodes


def is_side_effect_node(node: str) -> bool:
    return node in SIDE_EFFECT_NODES or node in {"tool", "artifact", "memory", "skill", "research"}


def should_delegate_to_odr(state: AgentRuntimeState) -> bool:
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    values = [
        route_plan.get("intent"),
        route_plan.get("research_mode"),
        state.get("intent"),
        state.get("route"),
        state.get("user_input"),
    ]
    for field in EXPLICIT_ACTION_FIELDS:
        values.append(state.get(field))
    if route_plan.get("explicit_research") or route_plan.get("intent") == "deep_research":
        return True
    text = " ".join(str(value or "").lower() for value in values)
    return any(term in text for term in DEEP_RESEARCH_TERMS)


def is_explicit_user_route(state: AgentRuntimeState) -> bool:
    return any(bool(state.get(field)) for field in EXPLICIT_ACTION_FIELDS)


def validate_and_normalize_llm_supervisor_route(
    decision: LLMSupervisorRouteDecision | dict[str, Any],
    state: AgentRuntimeState,
    *,
    available_nodes: set[str],
    planner_route: list[Any],
) -> NormalizedSupervisorRouteDecision:
    if isinstance(decision, dict):
        try:
            parsed = LLMSupervisorRouteDecision.model_validate(decision)
        except ValidationError as exc:
            return build_fallback_supervisor_decision(
                state,
                reason=f"invalid_llm_decision:{exc}",
                planner_route=planner_route,
                available_nodes=available_nodes,
            )
    else:
        parsed = decision

    if should_delegate_to_odr(state) or parsed.target_runtime == "odr":
        return _odr_decision(state, planner_route, raw_decision=parsed.model_dump())

    errors: list[str] = []
    route: list[str] = []
    for item in parsed.route:
        node = _normalize_node(str(item), available_nodes)
        if node is None:
            errors.append(f"unknown_node:{item}")
            continue
        route.append(node)
    route = _dedupe_route(route)

    if not route:
        fallback = build_fallback_supervisor_decision(
            state,
            reason="empty_or_unknown_llm_route",
            planner_route=planner_route,
            available_nodes=available_nodes,
        )
        fallback.validation_errors.extend(errors)
        fallback.raw_decision = parsed.model_dump()
        return fallback

    approval_status = str(state.get("approval_status") or "")
    if approval_status == "rejected":
        route = _safe_terminal_route(available_nodes)
        errors.append("approval_rejected_forced_final_response")

    planner_needs_approval = bool((state.get("route_plan") or {}).get("needs_approval"))
    has_side_effect = any(is_side_effect_node(node) for node in route)
    if "tool_agent" in route:
        requires_approval = bool(parsed.requires_approval or planner_needs_approval)
    else:
        requires_approval = bool(parsed.requires_approval)
    if has_side_effect and requires_approval and approval_status != "approved":
        route = _safe_terminal_route(available_nodes)
        errors.append("approval_pending_blocked_side_effect_route")

    route = _ensure_terminal(route, available_nodes)
    next_node = route[0] if route else None
    normalized = NormalizedSupervisorRouteDecision(
        target_runtime="web_app",
        route=route,
        next_node=next_node,
        agent_name=next_node,
        reason=parsed.reason or "LLM supervisor route decision.",
        risk_level=parsed.risk_level,
        requires_approval=requires_approval,
        confidence=parsed.confidence,
        validation_errors=errors,
        raw_decision=parsed.model_dump(),
        original_planner_route=list(planner_route),
        metadata=dict(parsed.metadata or {}),
    )
    return normalized


def build_fallback_supervisor_decision(
    state: AgentRuntimeState,
    *,
    reason: str,
    planner_route: list[Any] | None = None,
    available_nodes: set[str] | None = None,
) -> NormalizedSupervisorRouteDecision:
    available = available_nodes or get_available_runtime_nodes(state)
    planner = list(planner_route if planner_route is not None else _planner_route(state))

    if should_delegate_to_odr(state):
        decision = _odr_decision(state, planner, raw_decision=None)
        decision.fallback = True
        decision.fallback_reason = reason
        decision.metadata["fallback"] = True
        decision.metadata["fallback_reason"] = reason
        return decision

    if str(state.get("approval_status") or "") == "rejected":
        route = _safe_terminal_route(available)
    elif _route_is_valid(planner, available):
        route = [str(item) for item in planner]
    else:
        route = _intent_fallback_route(state, available)

    if not route:
        raise RuntimeError("no safe executable fallback node exists")

    next_node = route[0]
    return NormalizedSupervisorRouteDecision(
        target_runtime="web_app",
        route=route,
        next_node=next_node,
        agent_name=next_node,
        reason="Fallback supervisor route.",
        risk_level=_risk_l0_l3((state.get("route_plan") or {}).get("risk_level")),
        requires_approval=bool((state.get("route_plan") or {}).get("needs_approval", False)),
        fallback=True,
        fallback_reason=reason,
        original_planner_route=planner,
        metadata={"fallback": True, "fallback_reason": reason},
    )


def apply_supervisor_route_plan(
    state: AgentRuntimeState,
    decision: NormalizedSupervisorRouteDecision,
    *,
    mode: SupervisorMode,
) -> AgentRuntimeState:
    payload = decision.model_dump()
    state["llm_supervisor_decision"] = payload
    state["llm_supervisor_raw_response"] = decision.raw_decision
    state["llm_supervisor_validation_errors"] = list(decision.validation_errors)
    _append_trace(
        state,
        event="llm_supervisor_decision",
        mode=mode,
        route=decision.route,
        target_runtime=decision.target_runtime,
        overwritten=mode == "full",
        validation_errors=decision.validation_errors,
        fallback=decision.fallback,
        fallback_reason=decision.fallback_reason,
        explicit_override=decision.explicit_override,
    )

    if mode != "full":
        return state

    original_route_plan = dict(state.get("route_plan") or {})
    route_plan = {
        **original_route_plan,
        "intent": _intent_for_decision(state, decision),
        "route": list(decision.route),
        "risk_level": decision.risk_level,
        "needs_approval": bool(decision.requires_approval),
        "reason": decision.reason,
        "llm_supervisor": {
            "source": "llm_supervisor",
            "original_planner_route": decision.original_planner_route,
            "validation_errors": decision.validation_errors,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "metadata": decision.metadata,
        },
    }
    if decision.target_runtime == "odr":
        route_plan["intent"] = "research"
        route_plan["explicit_research"] = True
        route_plan["research_mode"] = "deep"
        route_plan["expected_output"] = route_plan.get("expected_output") or "research_report"

    if "tool_agent" not in decision.route:
        if route_plan.get("needs_approval") is True:
            route_plan["needs_approval"] = False
        if route_plan.get("answer_mode") == "tool_action":
            route_plan["answer_mode"] = "chat"
        if route_plan.get("expected_output") == "action_result":
            route_plan["expected_output"] = "answer"
        if str(route_plan.get("intent", "")).startswith("tool"):
            route_plan["intent"] = "chat"

    state["route_plan"] = route_plan
    state["execution_plan"] = execution_plan_from_route_plan(route_plan, state)
    state["route"] = "tool" if str(route_plan.get("intent", "")).startswith("tool.") else route_plan.get("intent", "chat")
    state["approval_required"] = bool(route_plan.get("needs_approval", False))
    state["answer_mode"] = route_plan.get("answer_mode", state.get("answer_mode", "chat"))
    state["llm_supervisor_original_route_plan"] = original_route_plan
    return state


async def _invoke_llm_supervisor(
    state: AgentRuntimeState,
    *,
    settings: LLMSupervisorSettings,
    available_nodes: set[str],
    planner_route: list[Any],
) -> LLMSupervisorRouteDecision:
    prompt = _build_user_prompt(state, available_nodes=available_nodes, planner_route=planner_route)
    if settings.model:
        model = get_chat_model_by_name(
            settings.model,
            temperature=settings.temperature,
            timeout_seconds=settings.timeout_seconds,
        )
    else:
        model = get_chat_model("planner", complexity="low", temperature=settings.temperature)

    async def _call() -> LLMSupervisorRouteDecision:
        if hasattr(model, "with_structured_output"):
            structured = model.with_structured_output(LLMSupervisorRouteDecision)
            if hasattr(structured, "ainvoke"):
                result = await structured.ainvoke([
                    ("system", WEB_APP_LLM_SUPERVISOR_SYSTEM_PROMPT),
                    ("user", prompt),
                ])
            else:
                result = structured.invoke([
                    ("system", WEB_APP_LLM_SUPERVISOR_SYSTEM_PROMPT),
                    ("user", prompt),
                ])
            if isinstance(result, LLMSupervisorRouteDecision):
                return result
            return LLMSupervisorRouteDecision.model_validate(result)

        message = await model.ainvoke(prompt) if hasattr(model, "ainvoke") else model.invoke(prompt)
        content = getattr(message, "content", message)
        payload = _parse_json(str(content))
        return LLMSupervisorRouteDecision.model_validate(payload)

    return await asyncio.wait_for(_call(), timeout=settings.timeout_seconds)


def _build_pre_rule_decision(
    state: AgentRuntimeState,
    available_nodes: set[str],
    planner_route: list[Any],
) -> NormalizedSupervisorRouteDecision | None:
    if should_delegate_to_odr(state):
        decision = _odr_decision(state, planner_route, raw_decision=None)
        decision.metadata["pre_rule"] = "deep_research_to_odr"
        return decision

    if str(state.get("approval_status") or "") == "rejected":
        route = _safe_terminal_route(available_nodes)
        next_node = route[0]
        return NormalizedSupervisorRouteDecision(
            target_runtime="web_app",
            route=route,
            next_node=next_node,
            agent_name=next_node,
            reason="User rejected approval; returning final response without side effects.",
            risk_level="L0",
            requires_approval=False,
            original_planner_route=list(planner_route),
            metadata={"pre_rule": "approval_rejected"},
        )

    if is_explicit_user_route(state):
        route = _route_from_explicit_state(state, available_nodes)
        if route:
            next_node = route[0]
            return NormalizedSupervisorRouteDecision(
                target_runtime="web_app",
                route=route,
                next_node=next_node,
                agent_name=next_node,
                reason="User explicitly selected this route/action.",
                risk_level=_risk_l0_l3((state.get("route_plan") or {}).get("risk_level")),
                requires_approval=any(is_side_effect_node(node) for node in route),
                explicit_override=True,
                original_planner_route=list(planner_route),
                metadata={"pre_rule": "explicit_user_route"},
            )
    return None


def _route_from_explicit_state(state: AgentRuntimeState, available_nodes: set[str]) -> list[str]:
    explicit_route = state.get("explicit_route")
    if isinstance(explicit_route, str):
        candidate = [item.strip() for item in explicit_route.split(",") if item.strip()]
    elif isinstance(explicit_route, list):
        candidate = [str(item) for item in explicit_route if item]
    else:
        candidate = []

    if not candidate:
        values = [state.get(field) for field in EXPLICIT_ACTION_FIELDS if field != "explicit_route"]
        text = " ".join(str(value or "").lower() for value in values)
        for key, node in ACTION_TO_NODE.items():
            if key in text:
                candidate = [node]
                break

    route = []
    for item in candidate:
        node = _normalize_node(str(item), available_nodes)
        if node:
            route.append(node)
    return _ensure_terminal(_dedupe_route(route), available_nodes)


def _odr_decision(
    state: AgentRuntimeState,
    planner_route: list[Any],
    *,
    raw_decision: dict[str, Any] | None,
) -> NormalizedSupervisorRouteDecision:
    available = get_available_runtime_nodes(state)
    route = ["research_agent"] if "research_agent" in available else _safe_terminal_route(available)
    route = _ensure_terminal(route, available)
    next_node = route[0]
    return NormalizedSupervisorRouteDecision(
        target_runtime="odr",
        route=route,
        next_node=next_node,
        agent_name=next_node,
        reason="Deep research delegates to the existing ODR adapter via research_agent.",
        risk_level="L1",
        requires_approval=False,
        raw_decision=raw_decision,
        original_planner_route=list(planner_route),
        metadata={"odr_adapter_entry": "research_agent"},
    )


def _build_user_prompt(
    state: AgentRuntimeState,
    *,
    available_nodes: set[str],
    planner_route: list[Any],
) -> str:
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    parallel = state.get("parallel_read_results") if isinstance(state.get("parallel_read_results"), dict) else {}
    rag_prepare = parallel.get("rag_prepare") if isinstance(parallel.get("rag_prepare"), dict) else {}

    supervisor_task: list[str] = [
        "Classify user_input independently. Do not assume planner_suggestion is correct.",
        "Compare your classification with planner_suggestion.intent and planner_suggestion.route.",
        "If they disagree, override the planner suggestion.",
        "If they agree, still verify that the route is safe, minimal, uses only available_executable_nodes, and matches evidence/approval readiness.",
    ]
    if "tool_agent" in available_nodes:
        supervisor_task.append("If the user does not ask for a tool or external side effect, do not route to tool_agent.")
    if "final_response" in available_nodes:
        supervisor_task.append("If the user only makes a casual statement or simple chat, route to final_response.")
    if "rag_agent" in available_nodes:
        supervisor_task.append("If the user asks about uploaded documents, attachments, or private knowledge, route to rag_agent.")
    if "research_agent" in available_nodes:
        supervisor_task.append("If the user asks for broad/current/open-ended research, route to research_agent.")
    if "memory_agent" in available_nodes:
        supervisor_task.append("If the user asks to remember a preference or fact, route to memory_agent.")
    supervisor_task.extend([
        "Never bypass permission or approval.",
        "Return only a JSON route decision. Do not answer the user.",
    ])

    payload = {
        "user_input": state.get("user_input") or state.get("query") or "",
        "available_executable_nodes": sorted(available_nodes),
        "planner_suggestion": {
            "intent": route_plan.get("intent") or state.get("route") or "chat",
            "route": planner_route,
            "risk_level": route_plan.get("risk_level", "L0"),
            "needs_approval": bool(route_plan.get("needs_approval") or state.get("approval_required")),
            "reason": route_plan.get("reason", ""),
        },
        "runtime_readiness": {
            "evidence_readiness": {
                "rag_evidence_count": len(context.get("rag_evidence") or []),
                "rag_prepare_evidence_count": len(rag_prepare.get("evidence") or []),
                "parallel_read_warnings": state.get("parallel_read_warnings", []),
            },
            "evidence_sufficient": bool(context.get("rag_evidence") or rag_prepare.get("evidence")),
            "approval_status": state.get("approval_status"),
            "risk_level": route_plan.get("risk_level", "L0"),
        },
        "explicit_user_action": {field: state.get(field) for field in EXPLICIT_ACTION_FIELDS if state.get(field)},
        "supervisor_task": supervisor_task,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_node(value: str, available_nodes: set[str]) -> str | None:
    if value in available_nodes:
        return value
    alias = NODE_ALIASES.get(value)
    if alias in available_nodes:
        return alias
    return None


def _route_is_valid(route: list[Any], available_nodes: set[str]) -> bool:
    return bool(route) and all(str(item) in available_nodes for item in route)


def _ensure_terminal(route: list[str], available_nodes: set[str]) -> list[str]:
    if not route:
        return _safe_terminal_route(available_nodes)
    if "evaluator" in available_nodes and "evaluator" not in route and route != ["final_response"]:
        if "final_response" in route:
            route.insert(route.index("final_response"), "evaluator")
        else:
            route.append("evaluator")
    if "final_response" in available_nodes and route[-1] != "final_response":
        route.append("final_response")
    return route


def _intent_fallback_route(state: AgentRuntimeState, available_nodes: set[str]) -> list[str]:
    intent = str((state.get("route_plan") or {}).get("intent") or state.get("route") or "chat")
    if intent in {"rag", "document_qa"} and "rag_agent" in available_nodes:
        return _ensure_terminal(["rag_agent"], available_nodes)
    return _safe_terminal_route(available_nodes)


def _safe_terminal_route(available_nodes: set[str]) -> list[str]:
    if "final_response" in available_nodes:
        return ["final_response"]
    if "evaluator" in available_nodes:
        return ["evaluator"]
    return []


def _dedupe_route(route: list[str]) -> list[str]:
    result: list[str] = []
    for item in route:
        if item not in result:
            result.append(item)
    return result


def _planner_route(state: AgentRuntimeState) -> list[Any]:
    route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
    return list(route_plan.get("route") or [])


def _intent_for_decision(state: AgentRuntimeState, decision: NormalizedSupervisorRouteDecision) -> str:
    current = str((state.get("route_plan") or {}).get("intent") or state.get("route") or "chat")
    if decision.target_runtime == "odr":
        return "research"
    if "tool_agent" in decision.route:
        return current if current.startswith("tool.") else "tool"
    if "artifact_agent" in decision.route:
        return "artifact"
    if "memory_agent" in decision.route:
        return "memory"
    if "skill_agent" in decision.route:
        return "skill"
    if "rag_agent" in decision.route:
        return "rag"
    return "chat"


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("LLM supervisor output must be a JSON object")
    return data


def _risk_l0_l3(value: Any) -> RiskLevel:
    risk = str(value or "L0")
    if risk in {"L0", "L1", "L2", "L3"}:
        return risk  # type: ignore[return-value]
    if risk == "L4":
        return "L3"
    return "L0"


def _append_trace(state: AgentRuntimeState, **event: Any) -> None:
    state.setdefault("llm_supervisor_trace", []).append(event)


def _bool_config(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
