import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.dispatch import (
    END_SENTINEL,
    after_permission,
    dispatch_next_route_node,
    legacy_next_route_node,
    map_route_to_node,
)
from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.agent.runtime.graph_builder import build_agent_runtime_graph
from src.web_app.agent.runtime.graph_registry import (
    FALLBACK_NODE_NAMES,
    GRAPH_NODE_NAMES,
    build_fallback_nodes,
    build_runtime_node_registry,
)
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.tests.db_test_utils import make_test_session


def _runtime():
    return AgentRuntime(make_test_session(), {})


def _state(route, completed=None, **extra):
    state = {
        "user_id": 1,
        "run_id": 1,
        "thread_id": "t",
        "route_plan": {"intent": "test", "route": route, "risk_level": "L1"},
        "completed_nodes": completed or [],
    }
    state.update(extra)
    return state


def test_runtime_node_registry_contains_graph_callables():
    nodes = RuntimeNodes(make_test_session(), {})
    registry = build_runtime_node_registry(nodes)

    assert tuple(registry) == GRAPH_NODE_NAMES
    assert all(callable(value) for value in registry.values())
    assert callable(registry["context_builder"])
    assert callable(registry["skill_matcher"])


def test_graph_builder_builds_compiled_graph_and_runtime_uses_facade_wrappers():
    runtime = _runtime()
    graph = build_agent_runtime_graph(
        runtime.nodes,
        after_permission=runtime._after_permission,
        dispatch_next_route_node=runtime._dispatch_next_route_node,
    )

    assert graph is not None
    assert callable(runtime._build_langgraph().ainvoke)
    assert list(inspect.signature(AgentRuntime._dispatch_next_route_node).parameters) == ["self", "state"]
    assert list(inspect.signature(AgentRuntime._after_permission).parameters) == ["self", "state"]


def test_graph_builder_source_keeps_p6a_wiring():
    text = (_ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_builder.py").read_text(encoding="utf-8")

    assert 'workflow.add_edge("planner", "parallel_prefetch")' in text
    assert 'workflow.add_edge("parallel_prefetch", "parallel_read_stage")' in text
    assert 'workflow.add_edge("parallel_read_stage", "supervisor_observer")' in text
    assert 'workflow.add_conditional_edges(\n        "supervisor_observer"' in text
    assert 'workflow.add_edge("parallel_prefetch", "context_builder")' not in text


def test_fallback_node_order_uses_parallel_read_and_supervisor_observer():
    runtime = _runtime()
    fallback_nodes = runtime._fallback_nodes()

    assert [node.__name__ for node in fallback_nodes] == list(FALLBACK_NODE_NAMES)
    assert fallback_nodes == build_fallback_nodes(runtime.nodes)
    assert "context_builder" not in [node.__name__ for node in fallback_nodes]
    assert "skill_matcher" not in [node.__name__ for node in fallback_nodes]


def test_after_permission_wrapper_matches_dispatch_helper():
    runtime = _runtime()
    blocked = {"route": "blocked"}
    clean = {}

    assert after_permission(blocked) == "done"
    assert runtime._after_permission(blocked) == "done"
    assert after_permission(clean) == "continue"
    assert runtime._after_permission(clean) == "continue"


def test_map_route_to_node_preserves_aliases_and_unknown_fallback():
    runtime = _runtime()

    assert map_route_to_node("rag_agent") == "rag_agent"
    assert runtime._map_route_to_node("rag_agent") == "rag_agent"
    assert map_route_to_node("rag") == "rag_agent"
    assert map_route_to_node("unknown_agent") == "final_response"


def test_dispatch_equivalence_with_legacy_cases():
    cases = [
        ("chat route", _state([]), "final_response"),
        ("rag route", _state(["rag_agent"]), "rag_agent"),
        ("tool route", _state(["tool_agent"]), "tool_agent"),
        ("artifact route", _state(["artifact_agent"]), "artifact_agent"),
        ("memory route", _state(["memory_agent"]), "memory_agent"),
        ("research route", _state(["research_agent"]), "research_agent"),
        ("waiting approval", _state(["tool_agent"], status="waiting_approval"), END_SENTINEL),
        ("completed route", _state(["rag_agent"], completed=["rag_agent"]), "final_response"),
        ("empty route", _state([]), "final_response"),
    ]

    runtime = _runtime()
    for _name, state, expected in cases:
        legacy_state = dict(state)
        dispatch_state = dict(state)

        assert legacy_next_route_node(legacy_state) == expected
        assert runtime._dispatch_next_route_node(dispatch_state) == expected
        assert dispatch_next_route_node(dict(state)) == expected


def test_dispatch_records_supervisor_observation_without_changing_route_state():
    runtime = _runtime()
    state = _state(["rag_agent"], supervisor_decision={"mode": "observe_only", "next_expected_node": "rag_agent"})
    route_plan = state["route_plan"]
    completed = state["completed_nodes"]

    next_node = runtime._dispatch_next_route_node(state)

    assert next_node == "rag_agent"
    assert state["route_plan"] is route_plan
    assert state["completed_nodes"] is completed
    assert state["supervisor_dispatch_audit"]["status"] == "ok"
    assert "supervisor_readiness_report" in state
    assert "supervisor_control_decision" in state
