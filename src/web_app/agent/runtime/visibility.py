"""Visibility Policy — decides which internal events become user-visible SSE.

Codex-style rule: internal events are always audited (DB), but only a small
subset is projected to the user-facing SSE stream.  This module is the single
place that encodes those rules.
"""

from typing import Any

# ── Nodes whose visible_thoughts are user-visible per intent ──────────
# Every *other* node still writes to DB but is NOT pushed to SSE.
_USER_VISIBLE_NODES: dict[str, set[str]] = {
    "chat": {
        "final_response",  # "我先根据当前上下文直接回答。"
    },
    "research": {
        "planner",
        "research_agent",
        "final_response",
    },
    "feed_research": {
        "planner",
        "research_agent",
        "final_response",
    },
    "rag": {
        "planner",
        "rag_agent",
        "final_response",
    },
    "artifact": {
        "planner",
        "artifact_agent",
        "final_response",
    },
    "tool": {
        "planner",
        "tool_agent",
        "final_response",
    },
    "memory": {
        "planner",
        "memory_agent",
        "final_response",
    },
    "skill": {
        "planner",
        "skill_agent",
        "final_response",
    },
    "mixed": {
        "planner",
        "research_agent",
        "rag_agent",
        "artifact_agent",
        "final_response",
    },
}


def _resolve_intent(state: dict[str, Any]) -> str:
    """Extract the canonical intent string from agent state."""
    route_plan = state.get("route_plan") or {}
    home_intent = state.get("home_intent") or {}
    return str(
        route_plan.get("intent")
        or home_intent.get("intent")
        or home_intent.get("detected_intent")
        or state.get("route")
        or "chat"
    )


def should_show_visible_thought_to_user(state: dict[str, Any], node_name: str) -> bool:
    """Return True if a visible_thought from *node_name* should be pushed to SSE.

    All visible_thoughts are still written to DB for audit regardless of this
    function's return value.
    """
    intent = _resolve_intent(state)
    allowed = _USER_VISIBLE_NODES.get(intent, _USER_VISIBLE_NODES.get("chat", set()))
    return node_name in allowed


def is_chat_fast_path(state: dict[str, Any]) -> bool:
    """Return True when the request qualifies for the chat fast-path.

    Fast-path chat shows at most 0-1 visible_progress entries and streams
    the answer directly, skipping all internal pipeline display.
    """
    intent = _resolve_intent(state)
    if intent != "chat":
        return False
    route_plan = state.get("route_plan") or {}
    home_intent = state.get("home_intent") or {}
    risk = str(route_plan.get("risk_level") or home_intent.get("risk_level") or "L0")
    if risk in ("L3", "L4"):
        return False
    return True


def user_visible_milestones_for_intent(intent: str) -> list[str]:
    """Return the ordered list of node names that may produce user-visible progress."""
    return list(_USER_VISIBLE_NODES.get(intent, _USER_VISIBLE_NODES.get("chat", [])))
