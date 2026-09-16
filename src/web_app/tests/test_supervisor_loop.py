"""Behavior contracts for the real compiled Supervisor loop."""

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.web_app.agent.runtime.graph_builder import build_graph
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.agent.runtime.state import SupervisorAction, CapabilityResult
from src.web_app.agent.runtime.capabilities import observe
from src.web_app.agent.runtime.context import research_consent
from src.web_app.agent.runtime.finalization import finish
from src.web_app.tests.test_chat_control import env as env


def state(env):
    return {
        "runtime_version": 2,
        "user_id": env.user,
        "run_id": env.run,
        "conversation_id": "chat",
        "thread_id": f"run:{env.run}",
        "user_input": "查一下资料",
        "interaction_version": 2,
    }


def configure(nodes, actions, monkeypatch):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    if isinstance(nodes, SupervisorNodes):
        from src.web_app.tests.test_native_supervisor import Model, call, configure as configure_native
        from langchain_core.messages import AIMessageChunk
        turns = []
        for action in actions:
            kind, args = action["action"], action.get("arguments", {})
            turns.append([AIMessageChunk(content="已完成")] if kind == "respond" else [
                call(args["name"] if kind == "tool" else kind, args.get("input", {}) if kind == "tool" else args)])
        class ScriptModel(Model):
            async def astream(self, messages):
                if not self.tools:
                    yield AIMessageChunk(content="Budget reached; existing results preserved.")
                    return
                async for chunk in super().astream(messages):
                    yield chunk
        configure_native(nodes, ScriptModel(turns), monkeypatch)
        return


@pytest.mark.asyncio
async def test_dynamic_loop_repeats_capability_without_route_plan(env, monkeypatch):
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(
            nodes,
            [{"action": "rag", "arguments": {"query": q}} for q in ["first", "refined"]]
            + [{"action": "respond"}],
            monkeypatch,
        )
        seen = []

        async def capability(s):
            assert "route_plan" not in s and "completed_nodes" not in s
            a = s["current_action"]
            seen.append((a["action_id"], a["arguments"]["query"]))
            return observe(
                nodes,
                s,
                CapabilityResult(
                    action_id=a["action_id"],
                    capability="rag",
                    status="empty" if len(seen) == 1 else "ok",
                ),
            )

        monkeypatch.setattr(nodes, "capability", capability)
        result = await build_graph(nodes).ainvoke(state(env))
        assert [q for _, q in seen] == ["first", "refined"]
        assert len({a for a, _ in seen}) == 2
        assert (
            result["status"] == "completed" and result["runtime_budget"]["steps"] == 3
        )


@pytest.mark.asyncio
async def test_tool_budget_is_harness_owned(env, monkeypatch):
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(
            nodes,
            [{"action": "tool", "arguments": {"name": "system.time"}}] * 10,
            monkeypatch,
        )

        async def capability(s):
            return s

        monkeypatch.setattr(nodes, "tool_runtime", capability)
        result = await build_graph(nodes).ainvoke(state(env), {"recursion_limit": 60})
        assert result["runtime_budget"]["tool_calls"] == 8
        assert result["termination_reason"] == "tool_budget_exhausted"


@pytest.mark.asyncio
async def test_unconfirmed_research_never_executes(env, monkeypatch):
    from src.web_app.services.research_service import research_service

    monkeypatch.setattr(
        research_service,
        "research_for_supervisor",
        lambda *a: pytest.fail("unconfirmed research"),
        raising=False,
    )
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(
            nodes,
            [
                {"action": "deep_research", "arguments": {"query": "topic"}},
                {
                    "action": "propose_deep_research",
                    "arguments": {
                        "query": "topic",
                        "benefit": "需要更多证据",
                        "question": "是否深入研究？",
                    },
                },
            ],
            monkeypatch,
        )
        result = await build_graph(nodes).ainvoke(state(env))
        assert result["observations"][0]["status"] == "blocked"
        assert result["research_proposal"]["query"] == "topic"
        assert result["status"] == "completed" and not result.get("approval_required")


@pytest.mark.parametrize(
    "text,proposal,allowed",
    [
        ("查一下最新情况", None, False),
        ("请深入研究这个问题", None, True),
        ("不要深入研究", None, False),
        ("好的", {"query": "topic"}, True),
        ("好的", None, False),
        ("考虑一下", {"query": "topic"}, False),
        ("换个话题", {"query": "topic"}, False),
    ],
)
def test_research_consent(text, proposal, allowed):
    assert bool(research_consent(text, {}, proposal)) is allowed


@pytest.mark.parametrize(
    "payload",
    [
        {
            "action": "tool",
            "arguments": {"name": "email.send", "requires_approval": False},
        },
        [{"action": "respond"}, {"action": "rag"}],
        {"action": "respond", "route": ["rag"]},
        {"action": "rag", "arguments": {"query": ""}},
    ],
)
def test_invalid_or_multiple_actions_rejected(payload):
    with pytest.raises(Exception):
        SupervisorAction.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approved", "rejected"])
async def test_real_interrupt_resume_does_not_reselect_or_duplicate(
    env, monkeypatch, decision
):
    from src.web_app.db.repositories.approval_repository import ApprovalRepository
    from src.web_app.db.repositories.mcp_repository import ToolCallRepository
    from src.web_app.mcp.local_provider import local_provider

    calls = []

    def provider(*args, **kwargs):
        calls.append(args)
        return {"success": True, "sent": True}

    monkeypatch.setattr(local_provider, "call", provider)
    saver = InMemorySaver()
    cfg = {"configurable": {"thread_id": f"run:{env.run}"}}
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(
            nodes,
            [
                {
                    "action": "tool",
                    "arguments": {
                        "name": "email.send",
                        "input": {
                            "to": "someone@example.test",
                            "subject": "subject",
                            "body": "body",
                        },
                    },
                }
            ],
            monkeypatch,
        )
        paused = await build_graph(nodes, saver).ainvoke(state(env), cfg)
        assert "__interrupt__" in paused, paused.get("observations")
        assert len(paused["__interrupt__"]) == 1 and calls == []
        payload = paused["__interrupt__"][0].value
        action_id = paused["current_action"]["action_id"]
        approval = ApprovalRepository(db).get_by_user(env.user, payload["approval_id"])
        ApprovalRepository(db).update(approval, status=decision)
    # New runtime/session, same checkpoint. No earlier model decision is replayed.
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(nodes, [{"action": "respond"}], monkeypatch)
        result = await build_graph(nodes, saver).ainvoke(
            Command(
                resume={"action": decision, "tool_call_id": payload["tool_call_id"]}
            ),
            cfg,
        )
        assert result["status"] == "completed"
        assert len(calls) == int(decision == "approved")
        assert result["observations"][0]["action_id"] == action_id
        assert result["runtime_budget"]["tool_calls"] == 1
        assert len(ToolCallRepository(db).list_by_user(env.user)) == 1


@pytest.mark.asyncio
async def test_capability_writes_require_explicit_intent(env, monkeypatch):
    from src.web_app.mcp.local_provider import local_provider

    monkeypatch.setattr(
        local_provider, "call", lambda *a, **kw: pytest.fail("unauthorized save")
    )
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        configure(
            nodes,
            [
                {
                    "action": "artifact",
                    "arguments": {"title": "report", "content": "content"},
                },
                {"action": "respond"},
            ],
            monkeypatch,
        )
        result = await build_graph(nodes).ainvoke(state(env))
        assert result["observations"][0]["error"] == "explicit_save_intent_required"


def test_api_defaults_are_not_explicit_save_authority():
    from src.web_app.agent.schemas import AgentRunRequest

    implicit = AgentRunRequest(user_input="查资料").runtime_payload()
    explicit = AgentRunRequest(
        user_input="查资料", save_artifact=True
    ).runtime_payload()
    assert "save_artifact" not in implicit["_explicit_fields"]
    assert "save_artifact" in explicit["_explicit_fields"]


def test_context_budget_preserves_required_input():
    from src.web_app.agent.runtime.context import bounded_prompt

    s = {"user_input": "latest", "context": {"huge": "测试 " * 2000}}
    prompt = bounded_prompt(s, "system", 400)
    assert "latest" in prompt and len(prompt) < 10000
    with pytest.raises(ValueError, match="context_budget"):
        bounded_prompt({"user_input": "test " * 1000}, "system", 20)


@pytest.mark.asyncio
async def test_model_unavailable_preserves_results_without_fallback(env, monkeypatch):
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})

        async def unavailable(s, **kwargs):
            raise TimeoutError("provider unavailable")

        monkeypatch.setattr(nodes, "model_turn", unavailable)
        s = state(env)
        await nodes.permission_guard(s)
        s["observations"] = [
            {
                "action_id": "a",
                "capability": "rag",
                "status": "ok",
                "summary": "已取得证据",
            }
        ]
        result = await nodes.supervisor(s)
        assert result["status"] == "failed"
        assert "已取得证据" in result["final_answer"]
        assert result["runtime_budget"]["steps"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", [False, True])
async def test_ledger_prevents_restream_after_checkpoint_crash(
    env, monkeypatch, completed
):
    from src.web_app.agent.runtime.finalization import emit
    from src.web_app.db.repositories.agent_repository import AgentEventRepository

    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = state(env)
        emit(nodes, s, "answer_started", {})
        emit(nodes, s, "answer_delta", {"text": "已有答案"})
        if completed:
            emit(nodes, s, "answer_completed", {"answer": "已有答案"})
        monkeypatch.setattr(
            nodes, "model_turn", lambda s, **kwargs: pytest.fail("must recover, not reselect")
        )
        result = await nodes.supervisor(s)
        assert result["final_answer"] == "已有答案"
        assert result["status"] == ("completed" if completed else "failed")
        events = AgentEventRepository(db).list_by_run(env.user, env.run)
        assert [e.event_type for e in events].count("answer_started") == 1
        assert [e.event_type for e in events].count("answer_completed") == 1


@pytest.mark.asyncio
async def test_second_approval_remains_paused_in_service(env, monkeypatch):
    from datetime import datetime
    from src.web_app.services.agent_service import _finalize_resume
    from src.web_app.models.orm import AgentRun
    from src.web_app.db.repositories.agent_repository import AgentEventRepository

    with env.factory() as db:
        s = {
            **state(env),
            "status": "waiting_approval",
            "approval_required": True,
            "pending_approval_id": "2",
            "approval_payload": {"approval_id": 2, "tool_call_id": 2},
        }
        result = await _finalize_resume(
            db=db,
            user_id=env.user,
            run_id=env.run,
            run=db.get(AgentRun, env.run),
            state=s,
            conversation_id="chat",
            thread_id=s["thread_id"],
            user_input=s["user_input"],
            started_at=datetime.now(),
            pause_mode="interrupt",
            pending_approval_id="1",
            pending_tool_call_id=1,
            pending_tool_name="email.send",
            stream_queue=asyncio.Queue(),
        )
        assert result["status"] == "waiting_approval"
        assert db.get(AgentRun, env.run).completed_at is None
        events = {
            e.event_type
            for e in AgentEventRepository(db).list_by_run(env.user, env.run)
        }
        assert "run_paused" in events and "approval_required" in events
        assert "answer_completed" not in events and "run_completed" not in events


@pytest.mark.asyncio
async def test_action_identity_reuses_receipt_but_new_action_can_repeat(
    env, monkeypatch
):
    from src.web_app.agent.runtime.tools import execute_tool
    from src.web_app.mcp.local_provider import local_provider

    calls = []
    monkeypatch.setattr(
        local_provider, "call", lambda *a, **kw: calls.append(1) or {"value": "now"}
    )
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = {**state(env), "current_action": {"action": "tool", "action_id": "first"}}
        first = await execute_tool(nodes, s, "system.time", {})
        again = await execute_tool(nodes, s, "system.time", {})
        s["current_action"]["action_id"] = "second"
        second = await execute_tool(nodes, s, "system.time", {})
        assert first.data["id"] == again.data["id"] != second.data["id"]
        assert calls == [1, 1]


@pytest.mark.asyncio
async def test_explicit_save_hook_reuses_same_business_receipt(env, monkeypatch):
    from src.web_app.agent.runtime.hooks import save_outputs
    from src.web_app.mcp.local_provider import local_provider

    calls = []
    monkeypatch.setattr(
        local_provider,
        "call",
        lambda *a, **kw: calls.append(1) or {"artifact_id": 42, "title": "report"},
    )
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = {
            **state(env),
            "save_policy": {"save_artifact": True},
            "runtime_budget": {"tool_calls": 0, "consecutive_failures": 0},
        }
        await save_outputs(nodes, s, "report")
        await save_outputs(nodes, s, "report")
        assert calls == [1]
        assert len(s["artifacts"]) == 1


@pytest.mark.asyncio
async def test_hook_failure_does_not_rewrite_successful_answer(env, monkeypatch):
    from src.web_app.agent.runtime.hooks import save_outputs
    from src.web_app.mcp.local_provider import local_provider

    def failure(*a, **kw):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(local_provider, "call", failure)
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = {
            **state(env),
            "save_policy": {"save_artifact": True},
            "runtime_budget": {"tool_calls": 0, "consecutive_failures": 0},
        }
        await save_outputs(nodes, s, "answer")
        result = finish(nodes, s, "answer")
        assert result["status"] == "completed" and result["final_answer"] == "answer"
        assert result["final_warnings"] and not result.get("artifacts")
