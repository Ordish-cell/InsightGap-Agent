"""Single-runtime execution, durable format boundaries, and retained history."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.tests.test_chat_control import env as env
from src.web_app.tests.test_supervisor_loop import configure, state


@pytest.mark.asyncio
async def test_default_entry_is_supervisor(env, monkeypatch):
    from src.web_app.core.config import settings
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", False)
    with env.factory() as db:
        runtime = AgentRuntime(db, {})
        assert isinstance(runtime.nodes, SupervisorNodes)
        configure(runtime.nodes, [{"action": "respond"}], monkeypatch)
        result = await runtime.run(state(env))
    assert result["runtime_version"] == 2
    assert result["loop_protocol_version"] == 1
    assert result["status"] == "completed"
    assert "route_plan" not in result


@pytest.mark.parametrize("version", [1, 0, None, "1", 3])
def test_caller_cannot_select_another_runtime(env, version):
    with env.factory() as db, pytest.raises(ValueError, match="RUNTIME_VERSION_UNSUPPORTED"):
        AgentRuntime(db, {"runtime_version": version})


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [{}, {"runtime_version": 1}, {"runtime_version": 3}])
async def test_checkpoint_format_checked_before_resume(env, monkeypatch, values):
    from src.web_app.core.config import settings
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", True)
    graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values=values)),
                            ainvoke=AsyncMock())
    with env.factory() as db:
        runtime = AgentRuntime(db, {})
        monkeypatch.setattr(runtime, "_build_langgraph", AsyncMock(return_value=graph))
        with pytest.raises(ValueError, match="RUNTIME_VERSION_UNSUPPORTED"):
            await runtime.resume_from_interrupt({"action": "approved"}, f"run:{env.run}")
    graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 1])
async def test_service_rejects_old_run_without_changing_status(env, version):
    from src.web_app.models.orm import AgentRun
    from src.web_app.services.agent_service import resume_run_after_approval
    with env.factory() as db:
        run = db.get(AgentRun, env.run)
        run.status = "waiting_approval"
        run.graph_state = {} if version is None else {"runtime_version": version}
        db.commit()
        with pytest.raises(ValueError, match="RUNTIME_VERSION_UNSUPPORTED"):
            await resume_run_after_approval(db, env.user, env.run)
        db.refresh(run)
        assert run.status == "waiting_approval"


def test_historical_answers_are_projected_without_business_fallback():
    from src.web_app.services.agent_service import build_user_facing_answer
    assert build_user_facing_answer({"runtime_version": 1, "final_answer": "历史报告"}) == "历史报告"
    assert build_user_facing_answer({"route_plan": {"intent": "research"}, "user_input": "研究"}) == ""


def test_retired_switch_and_modules_are_absent():
    from pathlib import Path
    from src.web_app.core.config import Settings
    import src.web_app.agent.runtime as runtime
    assert "chat_fast_path_enabled" not in Settings.model_fields
    assert "chat_document_path_enabled" not in Settings.model_fields
    assert not hasattr(SupervisorNodes, "chat_entry")
    assert not hasattr(SupervisorNodes, "decide")
    assert not hasattr(SupervisorNodes, "answer")
    assert "agent_runtime_v2_enabled" not in Settings.model_fields
    assert "agent_planner_llm_enabled" not in Settings.model_fields
    assert "agent_intent_llm_enabled" not in Settings.model_fields
    directory = Path(runtime.__file__).parent
    assert not (directory / "v2").exists()
    assert not (directory / "node_groups").exists()
    for name in ("chat_fast_path", "document_chat", "native_supervisor", "planner", "dispatch", "fallback", "supervisor", "replanner", "recovery", "llm_supervisor"):
        assert not (directory / f"{name}.py").exists()


@pytest.mark.asyncio
async def test_bootstrap_does_not_perform_semantic_retrieval(env, monkeypatch):
    from src.web_app.services.memory_service import memory_service
    from src.web_app.services.rag_service import rag_service
    monkeypatch.setattr(rag_service, "search_evidence", lambda *a, **k: pytest.fail("eager RAG"))
    monkeypatch.setattr(memory_service, "search_memory", lambda *a, **k: pytest.fail("eager semantic memory"))
    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        result = await nodes.bootstrap_context(state(env))
    assert "gssc_context" in result["context"]
    assert "runtime_budget" not in result  # bootstrap does not reset execution counters


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 0, 2, "1"])
async def test_pre_native_checkpoint_rejected_without_execution(env, monkeypatch, version):
    from src.web_app.core.config import settings
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", True)
    graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"runtime_version": 2, "loop_protocol_version": version})), ainvoke=AsyncMock())
    with env.factory() as db:
        runtime = AgentRuntime(db, {})
        monkeypatch.setattr(runtime, "_build_langgraph", AsyncMock(return_value=graph))
        with pytest.raises(ValueError, match="LOOP_PROTOCOL_UNSUPPORTED"):
            await runtime.resume_from_interrupt({"action": "approved"}, f"run:{env.run}")
    graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_service_rejects_pre_native_checkpoint_before_mutation(env):
    from src.web_app.models.orm import AgentRun, ToolCall
    from src.web_app.services.agent_service import resume_run_after_approval
    with env.factory() as db:
        run = db.get(AgentRun, env.run)
        run.status = "waiting_approval"
        run.graph_state = {"runtime_version": 2}
        db.commit()
        before = db.query(ToolCall).count()
        with pytest.raises(ValueError, match="LOOP_PROTOCOL_UNSUPPORTED"):
            await resume_run_after_approval(db, env.user, env.run)
        db.refresh(run)
        assert run.status == "waiting_approval" and db.query(ToolCall).count() == before
