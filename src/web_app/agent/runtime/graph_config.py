"""LangGraph invocation config helpers."""

from __future__ import annotations

from typing import Any

from src.web_app.agent.runtime.state import AgentRuntimeState


def build_langgraph_invoke_config(state: AgentRuntimeState) -> dict[str, Any]:
    """Build LangGraph config without mutating runtime state."""
    user_id = state.get("user_id")
    run_id = state.get("run_id")
    thread_id = state.get("thread_id") or f"user:{user_id}:run:{run_id}"
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "run_id": run_id,
        }
    }
