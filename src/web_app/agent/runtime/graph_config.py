"""LangGraph invocation config helpers."""

from __future__ import annotations

from typing import Any

from src.web_app.agent.runtime.state import AgentRuntimeState


def build_langgraph_invoke_config(state: AgentRuntimeState) -> dict[str, Any]:
    """Build LangGraph config without mutating runtime state.

    thread_id is the checkpoint key — must be stable across pause/resume.
    Primary source: state["thread_id"]. Fallback: "run:{run_id}".
    """
    run_id = state.get("run_id")
    thread_id = str(state.get("thread_id") or f"run:{run_id}")
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }
