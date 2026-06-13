import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.runnables import RunnableLambda

from src.web_app.agent.runtime.graph_registry import (
    GRAPH_NODE_NAMES,
    RUNTIME_NODE_SPECS,
)
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.runnables import (
    as_runnable_node,
    build_runtime_runnable_registry,
)
from src.web_app.core.config import settings
from src.web_app.tests.db_test_utils import make_test_session


def test_runnable_registry_contains_all_graph_nodes():
    registry = build_runtime_runnable_registry(RuntimeNodes(make_test_session(), {}))

    assert tuple(registry) == GRAPH_NODE_NAMES
    assert all(isinstance(value, RunnableLambda) for value in registry.values())
    assert all(spec.runnable_enabled for spec in RUNTIME_NODE_SPECS)


def test_legacy_fallback_nodes_are_not_in_runnable_registry():
    registry = build_runtime_runnable_registry(RuntimeNodes(make_test_session(), {}))

    legacy_only_names = {
        "research",
        "rag",
        "artifact",
        "skill_librarian",
        "tool",
        "memory_writer",
        "skill_draft_detector",
    }
    assert legacy_only_names.isdisjoint(registry)


@pytest.mark.asyncio
async def test_async_node_runnable_ainvoke_matches_direct_call():
    calls = []

    async def sample_node(state):
        calls.append(dict(state))
        result = dict(state)
        result["called"] = True
        return result

    state = {"user_id": 1, "run_id": 2}
    runnable = as_runnable_node("sample_node", sample_node)

    direct = await sample_node(state)
    via_runnable = await runnable.ainvoke(state)

    assert via_runnable == direct
    assert calls == [state, state]


def test_runnable_adapter_has_clear_missing_langchain_fallback(monkeypatch):
    import src.web_app.agent.runtime.runnables as runnable_module

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_core.runnables":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert runnable_module.build_runtime_runnable_registry(RuntimeNodes(make_test_session(), {})) == {}
    with pytest.raises(RuntimeError, match="RunnableLambda is unavailable"):
        runnable_module.as_runnable_node("x", lambda state: state)


def test_graph_builder_does_not_use_runnable_registry():
    text = (_ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_builder.py").read_text(encoding="utf-8")

    assert "build_runtime_node_registry" in text
    assert "build_runtime_runnable_registry" not in text
    assert "runnables" not in text


def test_p6b_checkpointer_default_remains_disabled():
    assert settings.agent_langgraph_checkpointer_enabled is False
