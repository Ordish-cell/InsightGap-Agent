import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import graph as runtime_graph
from src.web_app.agent.runtime import graph_builder as graph_builder_module
from src.web_app.agent.runtime import graph_registry
from src.web_app.agent.runtime.checkpointers import build_checkpointer
from src.web_app.agent.runtime.dispatch import map_route_to_node
from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.agent.runtime.graph_builder import build_agent_runtime_graph
from src.web_app.agent.runtime.graph_config import build_langgraph_invoke_config
from src.web_app.agent.runtime.graph_registry import (
    AGENT_NODE_NAMES,
    GRAPH_NODE_NAMES,
    ROUTE_DESTINATION_NODE_NAMES,
    RUNTIME_NODE_SPECS,
    RuntimeNodeSpec,
    build_runtime_node_registry,
)
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.tests.db_test_utils import make_test_session


def _runtime():
    return AgentRuntime(make_test_session(), {})


def test_langgraph_invoke_config_matches_legacy_inline_shape():
    state = {"user_id": 7, "run_id": 11, "thread_id": "thread-abc"}

    assert build_langgraph_invoke_config(state) == {
        "configurable": {
            "thread_id": "thread-abc",
            "user_id": 7,
            "run_id": 11,
        }
    }


def test_langgraph_invoke_config_keeps_thread_fallback_and_does_not_mutate_state():
    state = {"user_id": 7, "run_id": 11}
    before = dict(state)

    config = build_langgraph_invoke_config(state)

    assert config["configurable"]["thread_id"] == "user:7:run:11"
    assert config["configurable"]["user_id"] == 7
    assert config["configurable"]["run_id"] == 11
    assert state == before


def test_runtime_node_specs_preserve_registry_callable_output():
    nodes = RuntimeNodes(make_test_session(), {})
    registry = build_runtime_node_registry(nodes)

    assert all(isinstance(spec, RuntimeNodeSpec) for spec in RUNTIME_NODE_SPECS)
    assert tuple(registry) == GRAPH_NODE_NAMES
    assert all(callable(value) for value in registry.values())


def test_route_destination_specs_match_formal_dispatch_nodes():
    spec_destinations = {spec.name for spec in RUNTIME_NODE_SPECS if spec.is_route_destination}
    formal_dispatch_destinations = {
        map_route_to_node(name) for name in (*AGENT_NODE_NAMES, "evaluator", "final_response")
    }

    assert spec_destinations == set(ROUTE_DESTINATION_NODE_NAMES)
    assert spec_destinations == formal_dispatch_destinations
    assert "context_builder" not in spec_destinations
    assert "skill_matcher" not in spec_destinations


def test_graph_builder_default_does_not_pass_checkpointer(monkeypatch):
    captured = {}

    class FakeWorkflow:
        def __init__(self, state_type):
            self.state_type = state_type

        def add_node(self, *args, **kwargs):
            pass

        def set_entry_point(self, *args, **kwargs):
            pass

        def add_conditional_edges(self, *args, **kwargs):
            pass

        def add_edge(self, *args, **kwargs):
            pass

        def compile(self, **kwargs):
            captured["compile_kwargs"] = kwargs
            return "compiled"

    monkeypatch.setattr(graph_builder_module, "build_runtime_node_registry", lambda nodes: {})
    monkeypatch.setitem(sys.modules, "langgraph.graph", type("FakeLangGraph", (), {"END": "__end__", "StateGraph": FakeWorkflow}))

    result = build_agent_runtime_graph(
        RuntimeNodes(make_test_session(), {}),
        after_permission=lambda state: "continue",
        dispatch_next_route_node=lambda state: "final_response",
    )

    assert result == "compiled"
    assert captured["compile_kwargs"] == {}


def test_graph_builder_passes_checkpointer_only_when_supplied(monkeypatch):
    captured = {}

    class FakeWorkflow:
        def __init__(self, state_type):
            self.state_type = state_type

        def add_node(self, *args, **kwargs):
            pass

        def set_entry_point(self, *args, **kwargs):
            pass

        def add_conditional_edges(self, *args, **kwargs):
            pass

        def add_edge(self, *args, **kwargs):
            pass

        def compile(self, **kwargs):
            captured["compile_kwargs"] = kwargs
            return "compiled"

    checkpointer = object()
    monkeypatch.setattr(graph_builder_module, "build_runtime_node_registry", lambda nodes: {})
    monkeypatch.setitem(sys.modules, "langgraph.graph", type("FakeLangGraph", (), {"END": "__end__", "StateGraph": FakeWorkflow}))

    result = build_agent_runtime_graph(
        RuntimeNodes(make_test_session(), {}),
        after_permission=lambda state: "continue",
        dispatch_next_route_node=lambda state: "final_response",
        checkpointer=checkpointer,
    )

    assert result == "compiled"
    assert captured["compile_kwargs"] == {"checkpointer": checkpointer}


def test_agent_runtime_default_does_not_create_checkpointer(monkeypatch):
    monkeypatch.setattr(runtime_graph.settings, "agent_langgraph_checkpointer_enabled", False)
    monkeypatch.setattr(runtime_graph, "build_checkpointer", lambda redis_url=None: (_ for _ in ()).throw(AssertionError("should not create checkpointer")))

    graph = _runtime()._build_langgraph()

    assert graph is not None


def test_agent_runtime_creates_checkpointer_when_enabled(monkeypatch):
    captured = {}
    checkpointer = object()

    monkeypatch.setattr(runtime_graph.settings, "agent_langgraph_checkpointer_enabled", True)
    monkeypatch.setattr(runtime_graph.settings, "redis_url", "redis://example/0")

    def fake_checkpointer(redis_url=None):
        captured["redis_url"] = redis_url
        return checkpointer

    monkeypatch.setattr(runtime_graph, "build_checkpointer", fake_checkpointer)

    def fake_builder(nodes, *, after_permission, dispatch_next_route_node, checkpointer=None):
        captured["checkpointer"] = checkpointer
        return "compiled"

    monkeypatch.setattr(runtime_graph, "build_agent_runtime_graph", fake_builder)

    assert _runtime()._build_langgraph() == "compiled"
    assert captured["redis_url"] == "redis://example/0"
    assert captured["checkpointer"] is checkpointer


def test_checkpointer_fallback_logs_warning_for_missing_or_unavailable_redis(caplog):
    caplog.set_level(logging.WARNING, logger="src.web_app.agent.runtime.checkpointers")

    missing = build_checkpointer("")
    unavailable = build_checkpointer("redis://127.0.0.1:0/0")

    assert missing is not None
    assert unavailable is not None
    messages = [record.getMessage() for record in caplog.records]
    assert any("Redis checkpointer URL is empty" in message for message in messages)
    assert any("Redis checkpointer unavailable" in message for message in messages)


def test_agent_runtime_wrappers_remain_available():
    runtime = _runtime()

    assert callable(runtime._dispatch_next_route_node)
    assert callable(runtime._after_permission)
    assert callable(runtime._map_route_to_node)
