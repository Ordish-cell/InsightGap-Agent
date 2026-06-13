"""LangChain Runnable adapters for runtime nodes.

The main LangGraph path still uses raw RuntimeNodes callables. These adapters
provide a standard LangChain surface for tests, debugging, and future chains.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.web_app.agent.runtime.graph_registry import RUNTIME_NODE_SPECS
from src.web_app.agent.runtime.nodes import RuntimeNodes


def as_runnable_node(name: str, node_callable: Callable[..., Any]) -> Any:
    try:
        from langchain_core.runnables import RunnableLambda
    except Exception as exc:  # pragma: no cover - exercised only without langchain_core
        raise RuntimeError("langchain_core RunnableLambda is unavailable") from exc

    return RunnableLambda(node_callable, name=name)


def build_runtime_runnable_registry(nodes: RuntimeNodes) -> dict[str, Any]:
    try:
        from langchain_core.runnables import RunnableLambda  # noqa: F401
    except Exception:
        return {}

    registry: dict[str, Any] = {}
    for spec in RUNTIME_NODE_SPECS:
        if not spec.runnable_enabled:
            continue
        registry[spec.name] = as_runnable_node(spec.name, getattr(nodes, spec.attr_name))
    return registry
