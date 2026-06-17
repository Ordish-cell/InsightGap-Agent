import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime import nodes as runtime_nodes
from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.agent.runtime.nodes import RuntimeNodes
from src.web_app.agent.runtime.node_groups import agent_nodes
from src.web_app.agent.runtime.node_groups import eval_final_nodes
from src.web_app.agent.runtime.node_groups import final_helpers
from src.web_app.agent.runtime.node_groups import legacy_nodes
from src.web_app.agent.runtime.node_groups import read_nodes
from src.web_app.agent.runtime.node_groups import setup_nodes
from src.web_app.agent.runtime.node_groups import tool_helpers
from src.web_app.agent.runtime.node_groups.agent_nodes import AgentNodesMixin
from src.web_app.agent.runtime.node_groups.eval_final_nodes import EvalFinalNodesMixin
from src.web_app.agent.runtime.node_groups.legacy_nodes import LegacyNodesMixin
from src.web_app.agent.runtime.node_groups.read_nodes import ReadNodesMixin
from src.web_app.agent.runtime.node_groups.setup_nodes import SetupNodesMixin
from src.web_app.tests.db_test_utils import make_test_session


def test_runtime_nodes_facade_keeps_constructor_and_methods():
    signature = inspect.signature(RuntimeNodes)
    assert list(signature.parameters) == ["db", "payload"]

    required = [
        "permission_guard",
        "home_intent_react",
        "router",
        "planner",
        "parallel_prefetch",
        "parallel_read_stage",
        "supervisor_observer",
        "llm_supervisor_route",
        "context_builder",
        "skill_matcher",
        "research_agent",
        "rag_agent",
        "artifact_agent",
        "tool_agent",
        "memory_agent",
        "skill_agent",
        "evaluator",
        "final_response",
        "research",
        "rag",
        "artifact",
        "tool",
        "memory_writer",
        "skill_librarian",
        "skill_draft_detector",
    ]
    nodes = RuntimeNodes(make_test_session(), {})

    for name in required:
        assert callable(getattr(nodes, name))


def test_runtime_nodes_uses_split_mixins():
    assert issubclass(RuntimeNodes, SetupNodesMixin)
    assert issubclass(RuntimeNodes, ReadNodesMixin)
    assert issubclass(RuntimeNodes, AgentNodesMixin)
    assert issubclass(RuntimeNodes, EvalFinalNodesMixin)
    assert issubclass(RuntimeNodes, LegacyNodesMixin)


def test_graph_import_and_fallback_nodes_stay_compatible():
    runtime = AgentRuntime(make_test_session(), {})
    fallback_nodes = runtime._fallback_nodes()
    fallback_names = [node.__name__ for node in fallback_nodes]

    assert fallback_nodes
    assert all(callable(node) for node in fallback_nodes)
    assert fallback_names == [
        "permission_guard",
        "home_intent_react",
        "planner",
        "parallel_prefetch",
        "parallel_read_stage",
        "supervisor_observer",
        "llm_supervisor_route",
        "research",
        "rag",
        "artifact",
        "skill_librarian",
        "tool",
        "memory_writer",
        "skill_draft_detector",
        "evaluator",
        "final_response",
    ]
    assert runtime.nodes.__class__ is RuntimeNodes


def test_runtime_nodes_tool_monkeypatch_surface_propagates(monkeypatch):
    replacement = lambda *args, **kwargs: ("local_file.read", {"path": "README.md"})

    monkeypatch.setattr(runtime_nodes, "infer_tool", replacement)

    assert runtime_nodes.infer_tool is replacement
    assert agent_nodes.infer_tool is replacement
    assert legacy_nodes.infer_tool is replacement


def test_runtime_nodes_shared_helper_monkeypatch_surface_propagates(monkeypatch):
    record_step_replacement = lambda *args, **kwargs: None
    sanitize_replacement = lambda tool_name, args: {"tool_name": tool_name, "safe": True}

    monkeypatch.setattr(runtime_nodes, "record_step", record_step_replacement)
    monkeypatch.setattr(runtime_nodes, "_sanitize_tool_args", sanitize_replacement)

    assert setup_nodes.record_step is record_step_replacement
    assert read_nodes.record_step is record_step_replacement
    assert agent_nodes.record_step is record_step_replacement
    assert eval_final_nodes.record_step is record_step_replacement
    assert legacy_nodes.record_step is record_step_replacement
    assert agent_nodes._sanitize_tool_args is sanitize_replacement
    assert tool_helpers._sanitize_tool_args is sanitize_replacement


def test_runtime_nodes_repository_and_final_helper_monkeypatch_surface_propagates(monkeypatch):
    class FakeArtifactRepository:
        pass

    approval_context_replacement = lambda state: "approval context"

    monkeypatch.setattr(runtime_nodes, "ArtifactRepository", FakeArtifactRepository)
    monkeypatch.setattr(runtime_nodes, "_approval_context_line", approval_context_replacement)

    assert agent_nodes.ArtifactRepository is FakeArtifactRepository
    assert legacy_nodes.ArtifactRepository is FakeArtifactRepository
    assert eval_final_nodes._approval_context_line is approval_context_replacement
    assert final_helpers._approval_context_line is approval_context_replacement
