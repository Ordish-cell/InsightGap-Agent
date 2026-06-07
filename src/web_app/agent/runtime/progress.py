"""Natural-language progress templates for user-visible milestones.

These replace raw node-name labels with product-quality status lines.
They never expose internal node names, risk levels, or ReAct fields.
"""

from typing import Any


def progress_for_node(node_name: str, state: dict[str, Any]) -> str:
    """Return a single natural-language sentence describing what the agent
    is doing right now.  Only called for nodes that pass the visibility policy."""

    intent = _resolve_intent(state)

    # ── Shared / generic milestones ──────────────────────────────────
    if node_name == "planner":
        return _planner_progress(intent)
    if node_name == "final_response":
        return _final_response_progress(intent)

    # ── Intent-specific agent milestones ─────────────────────────────
    if node_name == "research_agent":
        return "我正在检索和筛选相关资料，优先保留能直接帮助你决策的内容。"
    if node_name == "rag_agent":
        return "我正在从你的知识库中检索相关内容，筛出能支撑回答的证据。"
    if node_name == "artifact_agent":
        return "我正在把前面的结果整理成可保存、可复用的产物。"
    if node_name == "tool_agent":
        if state.get("status") == "waiting_approval":
            return "这个操作需要审批，我已暂停等待你的确认。"
        return "我正在调用工具执行你请求的操作。"
    if node_name == "memory_agent":
        return "我正在判断这次对话中是否有值得保存的偏好或结论。"
    if node_name == "skill_agent":
        return "我正在判断这个流程是否值得沉淀为可复用技能。"

    return ""


def _planner_progress(intent: str) -> str:
    if intent in ("research", "feed_research"):
        return "我会先拆解问题并确定信息来源，然后逐步检索和验证。"
    if intent == "rag":
        return "我会先从你的知识库中检索相关信息，再组织回答。"
    if intent in ("artifact",):
        return "我会先理解你的需求，再生成结构化的产物。"
    if intent in ("tool",):
        return "我会先确认工具和参数，然后执行操作。"
    if intent in ("memory",):
        return "我会整理这次对话中的关键信息并写入记忆。"
    if intent in ("skill",):
        return "我会分析这次流程是否可以沉淀为可复用的技能。"
    if intent == "mixed":
        return "这是一个复合任务，我会拆解成多个步骤逐步完成。"
    return "我先根据当前上下文直接回答。"


def _final_response_progress(intent: str) -> str:
    if intent == "chat":
        return "我先根据当前上下文直接回答。"
    if intent in ("research", "feed_research"):
        return "我已经整理好研究结论，下面给出最终回答。"
    if intent == "rag":
        return "我已经整理好检索结果，下面给出最终回答。"
    if intent == "artifact":
        return "我已经生成产物，下面给出说明。"
    if intent == "tool":
        return "工具执行完成，下面给出结果说明。"
    return "我已经整理好结论，下面给出可以直接使用的回答。"


def _resolve_intent(state: dict[str, Any]) -> str:
    route_plan = state.get("route_plan") or {}
    home_intent = state.get("home_intent") or {}
    return str(
        route_plan.get("intent")
        or home_intent.get("intent")
        or home_intent.get("detected_intent")
        or state.get("route")
        or "chat"
    )
