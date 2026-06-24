import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.graph_manifest import (
    build_runtime_graph_manifest,
    render_runtime_graph_mermaid,
)
from src.web_app.agent.runtime.graph_registry import (
    FALLBACK_NODE_NAMES,
    GRAPH_NODE_NAMES,
    ROUTE_DESTINATION_NODE_NAMES,
    RUNTIME_NODE_SPECS,
)


def test_manifest_contains_all_graph_nodes_with_contract_metadata():
    manifest = build_runtime_graph_manifest()
    nodes = manifest["nodes"]

    assert [node["name"] for node in nodes] == list(GRAPH_NODE_NAMES)
    for node in nodes:
        assert node["kind"]
        assert isinstance(node["reads"], list)
        assert isinstance(node["writes"], list)
        assert node["side_effect_level"] in {"none", "read", "write", "external_tool", "final"}
        assert node["description"]


def test_runtime_node_specs_have_minimal_contract_fields():
    for spec in RUNTIME_NODE_SPECS:
        assert spec.description
        assert spec.side_effect_level in {"none", "read", "write", "external_tool", "final"}
        assert isinstance(spec.reads, tuple)
        assert isinstance(spec.writes, tuple)


def test_manifest_route_destinations_and_fallback_sequence_are_stable():
    manifest = build_runtime_graph_manifest()

    assert manifest["route_destinations"] == list(ROUTE_DESTINATION_NODE_NAMES)
    assert manifest["fallback_sequence"] == list(FALLBACK_NODE_NAMES)
    assert "parallel_prefetch" in manifest["fallback_sequence"]
    assert "parallel_read_stage" in manifest["fallback_sequence"]
    assert "supervisor_observer" in manifest["fallback_sequence"]
    assert "llm_supervisor_route" in manifest["fallback_sequence"]
    assert "context_builder" not in manifest["fallback_sequence"]
    assert "skill_matcher" not in manifest["fallback_sequence"]
    assert "replanner" not in manifest["fallback_sequence"]


def test_manifest_edges_match_current_graph_wiring():
    manifest = build_runtime_graph_manifest()
    edge_pairs = {(edge["from"], edge["to"]) for edge in manifest["edges"]}

    assert ("home_intent_react", "planner") in edge_pairs
    assert ("planner", "parallel_prefetch") in edge_pairs
    assert ("parallel_prefetch", "parallel_read_stage") in edge_pairs
    assert ("parallel_read_stage", "supervisor_observer") in edge_pairs
    assert ("supervisor_observer", "llm_supervisor_route") in edge_pairs
    assert ("llm_supervisor_route", "route_dispatch") in edge_pairs
    assert ("route_dispatch", "rag_agent") in edge_pairs
    assert ("route_dispatch", "evaluator") in edge_pairs
    assert ("evaluator", "recovery_dispatch") in edge_pairs
    assert ("recovery_dispatch", "final_response") in edge_pairs
    assert ("recovery_dispatch", "rag_agent") in edge_pairs
    assert "replanner" not in {edge["to"] for edge in manifest["edges"]}


def test_mermaid_render_includes_key_runtime_sections():
    mermaid = render_runtime_graph_mermaid()

    assert mermaid.startswith("flowchart TD")
    for token in (
        "planner",
        "parallel_prefetch",
        "parallel_read_stage",
        "supervisor_observer",
        "llm_supervisor_route",
        "route dispatch",
        "recovery dispatch",
        "replanner dispatch observer",
        "rag_agent --> route_dispatch",
        "END",
    ):
        assert token in mermaid


def test_manifest_contains_replanner_capabilities_without_adding_graph_node():
    manifest = build_runtime_graph_manifest()
    capabilities = manifest["replanner_capabilities"]

    assert capabilities["shadow_observation"] is True
    assert capabilities["candidate_plan"] is True
    assert capabilities["limited_control"] is True
    assert capabilities["control_default"] is False
    assert capabilities["control_allowlist"] == ["rag_agent", "final_response"]
    assert capabilities["dispatch_observer"] is True
    assert capabilities["graph_node"] is False
    assert "replanner" not in manifest["graph_node_names"]
    assert manifest["route_destinations"] == list(ROUTE_DESTINATION_NODE_NAMES)
    assert manifest["fallback_sequence"] == list(FALLBACK_NODE_NAMES)


def test_manifest_contains_evaluator_recovery_capabilities():
    manifest = build_runtime_graph_manifest()
    capabilities = manifest["evaluator_recovery_capabilities"]

    assert capabilities["evaluator_recovery_loop"] is True
    assert capabilities["direct_agent_retry"] is True
    assert capabilities["max_attempts_per_agent"] == 1
    assert capabilities["max_total_attempts"] == 6


def test_manifest_build_is_static_and_does_not_import_runtime_nodes():
    graph_manifest_path = _ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_manifest.py"
    text = graph_manifest_path.read_text(encoding="utf-8")

    assert "RuntimeNodes" not in text
    assert "build_runtime_node_registry" not in text
    assert "Session" not in text


def test_graph_builder_does_not_depend_on_manifest():
    graph_builder_path = _ROOT / "src" / "web_app" / "agent" / "runtime" / "graph_builder.py"
    text = graph_builder_path.read_text(encoding="utf-8")

    assert "graph_manifest" not in text
    assert "build_runtime_graph_manifest" not in text
    assert "render_runtime_graph_mermaid" not in text
