from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.web_app.agent.runtime.dispatch import dispatch_next_route_node
from src.web_app.agent.runtime.llm_supervisor import (
    LLMSupervisorRouteDecision,
    build_fallback_supervisor_decision,
    get_available_runtime_nodes,
    llm_supervisor_route_node,
    resolve_llm_supervisor_settings,
    validate_and_normalize_llm_supervisor_route,
)


def _state(route=None, intent="rag"):
    route = route if route is not None else ["rag_agent", "final_response"]
    return {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t1",
        "user_input": "hello",
        "route_plan": {
            "intent": intent,
            "route": list(route),
            "risk_level": "L1",
            "needs_approval": False,
            "answer_mode": "chat",
        },
        "completed_nodes": [],
    }


def _config(enabled=True, mode="full", model=""):
    return {
        "configurable": {
            "agent_llm_supervisor_enabled": enabled,
            "agent_llm_supervisor_mode": mode,
            "agent_llm_supervisor_model": model,
            "agent_llm_supervisor_temperature": 0,
            "agent_llm_supervisor_timeout_seconds": 2,
        }
    }


@pytest.mark.asyncio
async def test_disabled_does_not_call_llm_or_override(monkeypatch):
    called = False

    async def fake_call(*args, **kwargs):
        nonlocal called
        called = True
        return LLMSupervisorRouteDecision(route=["final_response"], reason="x")

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor._invoke_llm_supervisor", fake_call)
    state = _state()
    result = await llm_supervisor_route_node(state, config=_config(enabled=False, mode="full"))

    assert called is False
    assert result["route_plan"]["route"] == ["rag_agent", "final_response"]
    assert result["llm_supervisor_trace"][0]["event"] == "llm_supervisor_skipped"


@pytest.mark.asyncio
async def test_off_mode_does_not_call_llm_or_override(monkeypatch):
    called = False

    async def fake_call(*args, **kwargs):
        nonlocal called
        called = True
        return LLMSupervisorRouteDecision(route=["final_response"], reason="x")

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor._invoke_llm_supervisor", fake_call)
    state = _state()
    result = await llm_supervisor_route_node(state, config=_config(enabled=True, mode="off"))

    assert called is False
    assert result["route_plan"]["route"] == ["rag_agent", "final_response"]


@pytest.mark.asyncio
async def test_shadow_mode_records_llm_but_preserves_planner_route(monkeypatch):
    async def fake_call(*args, **kwargs):
        return LLMSupervisorRouteDecision(route=["final_response"], reason="chat")

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor._invoke_llm_supervisor", fake_call)
    state = _state()
    result = await llm_supervisor_route_node(state, config=_config(enabled=True, mode="shadow"))

    assert result["route_plan"]["route"] == ["rag_agent", "final_response"]
    assert result["llm_supervisor_decision"]["route"] == ["final_response"]
    assert dispatch_next_route_node(result) == "rag_agent"


@pytest.mark.asyncio
async def test_full_mode_overrides_route_plan_and_dispatch(monkeypatch):
    async def fake_call(*args, **kwargs):
        return LLMSupervisorRouteDecision(route=["final_response"], reason="direct")

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor._invoke_llm_supervisor", fake_call)
    state = _state()
    result = await llm_supervisor_route_node(state, config=_config(enabled=True, mode="full"))

    assert result["route_plan"]["route"] == ["final_response"]
    assert result["route_plan"]["llm_supervisor"]["original_planner_route"] == ["rag_agent", "final_response"]
    assert dispatch_next_route_node(result) == "final_response"


def test_config_model_overrides_env(monkeypatch):
    monkeypatch.setattr(
        "src.web_app.agent.runtime.llm_supervisor.get_settings",
        lambda: SimpleNamespace(
            agent_llm_supervisor_enabled=True,
            agent_llm_supervisor_mode="shadow",
            agent_llm_supervisor_model="env-model",
            agent_llm_supervisor_temperature=0,
            agent_llm_supervisor_timeout_seconds=20,
        ),
    )

    settings = resolve_llm_supervisor_settings(_config(enabled=True, mode="full", model="config-model"))

    assert settings.model == "config-model"
    assert settings.mode == "full"


@pytest.mark.asyncio
async def test_model_name_is_passed_to_factory(monkeypatch):
    seen = {}

    class FakeStructured:
        async def ainvoke(self, messages):
            return LLMSupervisorRouteDecision(route=["final_response"], reason="ok")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructured()

    def fake_by_name(model, **kwargs):
        seen["model"] = model
        return FakeModel()

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor.get_chat_model_by_name", fake_by_name)
    state = _state()

    await llm_supervisor_route_node(state, config=_config(enabled=True, mode="full", model="test-model-from-config"))

    assert seen["model"] == "test-model-from-config"


def test_unknown_node_is_filtered_and_route_remains_executable():
    state = _state()
    decision = validate_and_normalize_llm_supervisor_route(
        LLMSupervisorRouteDecision(route=["unknown_agent", "final_response"], reason="x"),
        state,
        available_nodes=get_available_runtime_nodes(state),
        planner_route=["rag_agent", "final_response"],
    )

    assert decision.route == ["final_response"]
    assert "unknown_node:unknown_agent" in decision.validation_errors


def test_empty_route_falls_back_to_planner_route():
    state = _state()
    decision = validate_and_normalize_llm_supervisor_route(
        LLMSupervisorRouteDecision(route=[], reason="x"),
        state,
        available_nodes=get_available_runtime_nodes(state),
        planner_route=["rag_agent", "final_response"],
    )

    assert decision.route == ["rag_agent", "final_response"]
    assert decision.fallback is True


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_planner_route(monkeypatch):
    async def fake_call(*args, **kwargs):
        raise TimeoutError("slow")

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor._invoke_llm_supervisor", fake_call)
    state = _state()
    result = await llm_supervisor_route_node(state, config=_config(enabled=True, mode="full"))

    assert result["route_plan"]["route"] == ["rag_agent", "final_response"]
    assert result["llm_supervisor_decision"]["fallback"] is True


def test_side_effect_approval_not_approved_blocks_new_side_effect_route():
    state = _state(route=[], intent="chat")
    state["route_plan"]["needs_approval"] = True
    decision = validate_and_normalize_llm_supervisor_route(
        LLMSupervisorRouteDecision(
            route=["tool_agent", "final_response"],
            reason="tool",
            risk_level="L3",
            requires_approval=True,
        ),
        state,
        available_nodes=get_available_runtime_nodes(state),
        planner_route=[],
    )

    assert "tool_agent" not in decision.route
    assert decision.route == ["final_response"]


@pytest.mark.asyncio
async def test_deep_research_uses_research_agent_without_llm(monkeypatch):
    called = False

    async def fake_call(*args, **kwargs):
        nonlocal called
        called = True
        return LLMSupervisorRouteDecision(route=["final_response"], reason="x")

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor._invoke_llm_supervisor", fake_call)
    state = _state(route=["final_response"], intent="research")
    state["route_plan"]["explicit_research"] = True
    result = await llm_supervisor_route_node(state, config=_config(enabled=True, mode="full"))

    assert called is False
    assert result["route_plan"]["route"] == ["research_agent", "evaluator", "final_response"]
    assert result["route_plan"]["explicit_research"] is True


@pytest.mark.asyncio
async def test_explicit_artifact_bypasses_llm(monkeypatch):
    called = False

    async def fake_call(*args, **kwargs):
        nonlocal called
        called = True
        return LLMSupervisorRouteDecision(route=["final_response"], reason="x")

    monkeypatch.setattr("src.web_app.agent.runtime.llm_supervisor._invoke_llm_supervisor", fake_call)
    state = _state()
    state["user_clicked_action"] = "artifact"
    result = await llm_supervisor_route_node(state, config=_config(enabled=True, mode="full"))

    assert called is False
    assert result["route_plan"]["route"] == ["artifact_agent", "evaluator", "final_response"]
    assert result["llm_supervisor_decision"]["explicit_override"] is True


def test_dispatcher_ignores_raw_llm_response_and_reads_route_plan():
    state = _state(route=["rag_agent", "final_response"])
    state["llm_supervisor_raw_response"] = {"route": ["final_response"]}

    assert dispatch_next_route_node(state) == "rag_agent"


def test_fallback_uses_safe_route_when_planner_route_is_invalid():
    state = _state(route=["unknown_agent"], intent="chat")
    decision = build_fallback_supervisor_decision(
        state,
        reason="test",
        planner_route=["unknown_agent"],
        available_nodes=get_available_runtime_nodes(state),
    )

    assert decision.route == ["final_response"]
