"""Connection ownership and unrecoverable approval prevention."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.web_app.tests.test_chat_control import env as env


def test_event_ledger_serializes_nested_timestamps(env):
    from datetime import datetime, timezone
    from src.web_app.agent.runtime.event_ledger import publish_event
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with env.factory() as db:
        event = publish_event(db, None, env.run, "tool_call_completed", {"output": {"created_at": stamp}}, user_id=env.user)
        assert event.payload_json["output"]["created_at"] == stamp.isoformat()


@pytest.mark.asyncio
async def test_v2_rejects_unsupported_redis_without_fallback(env, monkeypatch):
    from src.web_app.agent.runtime.graph import AgentRuntime
    from src.web_app.core.config import settings
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", True)
    monkeypatch.setattr(settings, "agent_checkpointer_backend", "redis")
    with env.factory() as db:
        with pytest.raises(RuntimeError, match="does not support.*Redis"):
            await AgentRuntime(db, {"runtime_version": 2})._build_langgraph()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_actual_stream_preserves_partial_and_finalizes_once(
    env, monkeypatch, fails
):
    from langchain_core.messages import AIMessageChunk
    from src.web_app.agent.runtime import nodes as module
    from src.web_app.db.repositories.agent_repository import AgentEventRepository
    from src.web_app.tests.test_supervisor_loop import state

    closed = []

    async def stream(*args, **kwargs):
        try:
            yield AIMessageChunk(content="已经取得的内容")
            if fails:
                raise RuntimeError("provider disconnected")
            yield AIMessageChunk(content="。")
        finally:
            closed.append(True)

    monkeypatch.setattr(
        module, "get_chat_model", lambda *a, **kw: SimpleNamespace(astream=stream, bind_tools=lambda tools: SimpleNamespace(astream=stream))
    )
    with env.factory() as db:
        nodes = module.SupervisorNodes(db, {})
        s = state(env)
        await nodes.permission_guard(s)
        result = await nodes.supervisor(s)
        assert result["status"] == ("failed" if fails else "completed")
        assert result["final_answer"].startswith("已经取得的内容")
        assert closed == [True]
        replay = await nodes.supervisor(state(env))
        assert replay["final_answer"] == result["final_answer"]
        assert closed == [True]
        events = AgentEventRepository(db).list_by_run(env.user, env.run)
        types = [e.event_type for e in events]
        assert types.count("answer_started") == types.count("answer_completed") == 1
        assert types.count("agent_text_delta") == (1 if fails else 2)
        assert types.count("answer_delta") == 1


@pytest.mark.asyncio
async def test_l3_without_checkpoint_creates_no_approval(env, monkeypatch):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.runtime.tools import execute_tool
    from src.web_app.services.mcp_service import mcp_service
    from src.web_app.tests.test_supervisor_loop import state

    monkeypatch.setattr(
        mcp_service,
        "call_tool",
        lambda *a, **kw: pytest.fail("created unrecoverable approval"),
    )
    with env.factory() as db:
        s = state(env)
        s["current_action"] = {"action": "tool", "action_id": "email"}
        result = await execute_tool(
            SupervisorNodes(db, {}),
            s,
            "email.send",
            {
                "to": "test@example.test",
                "subject": "test",
                "body": "test",
            },
        )
        assert result.status == "blocked"
        assert result.error == "approval_checkpoint_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_runtime_closes_owned_connection_on_failure(env, monkeypatch, resume):
    from src.web_app.agent.runtime.graph import AgentRuntime
    from src.web_app.core.config import settings
    from src.web_app.tests.test_supervisor_loop import state

    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", True)
    context = SimpleNamespace(__aexit__=AsyncMock())
    with env.factory() as db:
        runtime = AgentRuntime(db, {"runtime_version": 2})

        async def build():
            runtime._checkpointer = SimpleNamespace(_checkpointer_ctx=context)
            return SimpleNamespace(
                aget_state=AsyncMock(return_value=SimpleNamespace(values={"runtime_version": 2, "loop_protocol_version": 1})),
                ainvoke=AsyncMock(side_effect=RuntimeError("invoke failed"))
            )

        monkeypatch.setattr(runtime, "_build_langgraph", build)
        with pytest.raises(RuntimeError, match="invoke failed"):
            if resume:
                await runtime.resume_from_interrupt({"action": "approve"}, "run:test")
            else:
                await runtime.run(state(env))
        context.__aexit__.assert_awaited_once_with(None, None, None)
        assert runtime._checkpointer is None


@pytest.mark.asyncio
async def test_saver_setup_failure_closes_connection(monkeypatch):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from src.web_app.agent.runtime.checkpointers import _AsyncPostgresSaverHandle

    saver = SimpleNamespace(setup=AsyncMock(side_effect=RuntimeError("setup failed")))
    context = SimpleNamespace(
        __aenter__=AsyncMock(return_value=saver), __aexit__=AsyncMock()
    )
    monkeypatch.setattr(AsyncPostgresSaver, "from_conn_string", lambda *a: context)
    with pytest.raises(RuntimeError, match="setup failed"):
        await _AsyncPostgresSaverHandle.create("unused")
    context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "observations,tool_calls,expected",
    [
        ([{"capability": "skill", "data": {"operation": "match"}}], [], 1),
        ([{"capability": "skill", "data": {}}], [], 0),
        ([], [{"tool_name": "skill_mcp.create_draft", "status": "failed"}], 0),
    ],
)
async def test_skill_hook_distinguishes_matching_and_creation(
    env, monkeypatch, observations, tool_calls, expected
):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.runtime.hooks import save_outputs
    from src.web_app.services.skill_service import skill_service
    from src.web_app.mcp.local_provider import local_provider
    from src.web_app.tests.test_supervisor_loop import state

    calls = []
    monkeypatch.setattr(
        skill_service, "evaluate_reusability", lambda s: {"should_create": True}
    )
    monkeypatch.setattr(
        local_provider,
        "call",
        lambda *a, **kw: calls.append(1) or {"skill_id": 1, "skill": {"id": 1}},
    )
    with env.factory() as db:
        s = {
            **state(env),
            "save_policy": {"create_skill_draft": True},
            "runtime_budget": {"tool_calls": 0, "consecutive_failures": 0},
            "observations": observations,
            "tool_calls": tool_calls,
        }
        await save_outputs(SupervisorNodes(db, {}), s, "answer")
        assert len(calls) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,inputs",
    [
        ("artifact_mcp.create_text_artifact", {"title": "title"}),
        ("memory_mcp.add", {"content": ""}),
        ("skill_mcp.create_draft", {}),
    ],
)
async def test_generic_tool_alias_cannot_bypass_save_schema(
    env, monkeypatch, name, inputs
):
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.agent.runtime.tools import execute_tool
    from src.web_app.services.mcp_service import mcp_service
    from src.web_app.tests.test_supervisor_loop import state

    monkeypatch.setattr(
        mcp_service, "call_tool", lambda *a, **kw: pytest.fail("invalid save executed")
    )
    with env.factory() as db:
        s = {
            **state(env),
            "current_action": {"action": "tool", "action_id": "save"},
            "save_policy": {
                "save_artifact": True,
                "write_memory": True,
                "create_skill_draft": True,
            },
        }
        result = await execute_tool(SupervisorNodes(db, {}), s, name, inputs)
        assert result.error in {"missing_fields", "invalid_tool_arguments"}
