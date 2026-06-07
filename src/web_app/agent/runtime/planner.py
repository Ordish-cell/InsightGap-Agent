"""Planner Agent — deterministic rule-based routing for the multi-agent Supervisor.

No LLM dependency in the first version. Produces a RoutePlan that the
Supervisor graph uses for conditional routing.
"""

from typing import Any

from src.web_app.agent.runtime.intent_schema import normalize_agent_name
from src.web_app.agent.runtime.state import AgentIntent, AgentRuntimeState, RiskLevel, RoutePlan

# ── Keyword sets for intent detection ──────────────────────────────

_RESEARCH_TERMS = [
    "研究", "深挖", "分析", "报告", "为什么", "趋势", "机会", "信息差",
    "深度研究", "调研", "洞察", "前景", "对比", "竞品", "行业",
    "research", "deep research", "deep_research", "analyze", "trend",
]

_RAG_TERMS = [
    "根据文档", "上传", "文档", "知识库", "检索", "引用", "资料",
    "我的文档", "查一下", "找一下", "问答", "基于资料",
    "rag", "RAG", "vector", "qdrant", "总结重点", "整理要点",
]

_ARTIFACT_TERMS = [
    "生成文档", "方案", "网站", "页面", "代码", "图片提示词",
    "artifact", "生成一份", "写一份", "起草", "草稿", "文档",
    "产品方案", "MVP", "技术方案", "spec", "报告", "总结报告",
    "markdown", "网页", "prototype",
]

_TOOL_TERMS = [
    "发邮件", "发送邮件", "邮件", "评论", "操作", "打开", "点击",
    "提交", "发布", "删除", "修改外部系统", "付款", "支付", "转账",
    "email", "send", "post", "submit", "delete", "payment",
    "执行", "运行命令",
]

_MEMORY_TERMS = [
    "记住", "以后", "我的偏好", "长期", "记忆", "保存下来",
    "下次记住", "别忘了", "记录一下",
    "remember", "save preference", "memory",
]

_SKILL_TERMS = [
    "复用", "下次复用", "工作流", "skill", "技能", "自动化流程",
    "沉淀", "做成模板", "标准化", "重复做", "自动化",
    "create skill", "reusable", "workflow", "流程",
]

# ── Public entry point ──────────────────────────────────────────────


def plan_route(
    user_input: str,
    feed_card_id: int | str | None = None,
    matched_skills: list[dict[str, Any]] | None = None,
    context_summary: str = "",
    forced_route: str | None = None,
    forced_intent: str | None = None,
    home_intent: dict[str, Any] | None = None,
) -> RoutePlan:
    """Produce a RoutePlan from user_input and optional context.
    Rule-based, deterministic. No LLM.
    """
    text = user_input.lower()
    route: list[str] = []
    risk_level: RiskLevel = "L0"
    needs_approval = False
    intent: AgentIntent = "chat"
    reasons: list[str] = []

    # ── Respect forced route from payload (backward compat) ─────
    forced = forced_route or forced_intent
    if forced in ("research", "rag", "artifact", "skill", "memory", "tool"):
        intent = forced  # type: ignore[assignment]
        reasons.append(f"forced_route={forced}")

    # ── Detect intent ───────────────────────────────────────────
    is_research = any(term in text for term in _RESEARCH_TERMS)
    is_rag = any(term in text for term in _RAG_TERMS)
    is_artifact = any(term in text for term in _ARTIFACT_TERMS)
    is_tool = any(term in text for term in _TOOL_TERMS) or any(term in text for term in ["发邮件", "发送邮件", "邮件", "评论", "发布", "提交表单", "打开网页", "删除", "支付", "付款", "转账"])
    is_memory = any(term in text for term in _MEMORY_TERMS)
    is_skill = any(term in text for term in _SKILL_TERMS)

    llm_intent = str((home_intent or {}).get("intent") or (home_intent or {}).get("detected_intent") or "")
    if not forced and llm_intent in {"chat", "research", "rag", "artifact", "feed_research", "tool", "tool.email", "tool.browser", "tool.comment", "tool.form_submit", "memory", "skill", "mixed"}:
        intent = llm_intent  # type: ignore[assignment]
        reasons.append("home_intent_used")
        if llm_intent.startswith("tool."):
            is_tool = True

    # Feed card deep-dive takes priority
    if feed_card_id and is_research:
        intent = "feed_research"
        reasons.append("feed_card_deep_dive")
    elif feed_card_id:
        intent = "feed_research"
        reasons.append("feed_card_attached")

    # Determine primary intent
    if intent == "chat":
        if is_tool:
            intent = "tool"
            reasons.append("tool_keywords_detected")
        elif (is_research and is_artifact) or (is_rag and is_artifact):
            intent = "mixed"
            reasons.append("mixed_keywords_detected")
        elif is_research:
            intent = "research"
            reasons.append("research_keywords_detected")
        elif is_artifact:
            intent = "artifact"
            reasons.append("artifact_keywords_detected")
        elif is_rag:
            intent = "rag"
            reasons.append("rag_keywords_detected")
        elif is_skill:
            intent = "skill"
            reasons.append("skill_keywords_detected")
        elif is_memory:
            intent = "memory"
            reasons.append("memory_keywords_detected")

    # ── Build route ─────────────────────────────────────────────
    # Route only contains nodes AFTER skill_matcher in the graph.
    # permission_guard, planner, context_builder, skill_matcher are
    # hard-wired in the graph — they always run.

    # Core agent nodes based on intent
    if intent in ("research", "feed_research", "mixed"):
        route.append("research_agent")
        reasons.append("research_agent_in_route")
    if intent in ("rag", "mixed") or is_rag:
        route.append("rag_agent")
        reasons.append("rag_agent_in_route")
    if intent in ("artifact", "mixed") or is_artifact:
        route.append("artifact_agent")
        reasons.append("artifact_agent_in_route")
    if intent == "tool" or str(intent).startswith("tool.") or is_tool:
        route.append("tool_agent")
        reasons.append("tool_agent_in_route")

    # Post-execution: memory/skill BEFORE evaluator (so evaluator can score them)
    if intent not in ("chat",):
        route.append("memory_agent")
    if is_skill or intent == "skill":
        route.append("skill_agent")

    # evaluator always before final_response
    route.append("evaluator")

    # final_response is always last
    route.append("final_response")

    # ── Risk assessment ─────────────────────────────────────────
    if is_tool:
        high_risk = any(t in text for t in ["删除", "支付", "付款", "转账", "delete", "payment"])
        ext_write = any(t in text for t in ["发邮件", "发送邮件", "评论", "发布", "提交", "email", "send", "post", "submit"])
        high_risk = high_risk or any(t in text for t in ["删除", "支付", "付款", "转账", "delete", "payment"])
        ext_write = ext_write or any(t in text for t in ["发邮件", "发送邮件", "邮件", "评论", "发布", "提交", "email", "send", "post", "submit"])
        if high_risk:
            risk_level = "L4"
            needs_approval = True
            reasons.append("high_risk_l4")
        elif ext_write:
            risk_level = "L3"
            needs_approval = True
            reasons.append("external_write_l3")
        elif intent == "tool":
            risk_level = "L2"
    elif intent in ("research", "feed_research", "rag"):
        risk_level = "L1"
    elif intent == "artifact":
        risk_level = "L2"

    if home_intent:
        llm_risk = str(home_intent.get("risk_level") or "L0")
        if llm_risk in {"L0", "L1", "L2", "L3", "L4"}:
            risk_level = _max_risk(risk_level, llm_risk)  # type: ignore[arg-type]
        needs_approval = needs_approval or risk_level in {"L3", "L4"} or bool(home_intent.get("needs_approval"))
        route = _merge_route(route, home_intent.get("suggested_route_hints") or home_intent.get("required_agents") or [])
        reasons.append("risk_guard_applied")

    # Expected output
    expected_output_map = {
        "research": "research_report",
        "feed_research": "research_report",
        "rag": "answer_with_evidence",
        "artifact": "document_or_code",
        "tool": "action_result",
        "skill": "skill_draft",
        "memory": "memory_update",
        "mixed": "structured_report",
        "chat": "answer",
    }

    route_plan: RoutePlan = {
        "intent": intent,
        "route": route,
        "risk_level": risk_level,
        "needs_approval": needs_approval,
        "expected_output": expected_output_map.get(intent, "answer"),
        "reason": "; ".join(reasons) if reasons else "default_chat_route",
    }
    return route_plan


def _max_risk(left: RiskLevel, right: str) -> RiskLevel:
    order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
    return left if order[left] >= order.get(right, 0) else right  # type: ignore[return-value]


def _merge_route(rule_route: list[str], hints: Any) -> list[str]:
    executable = {"research_agent", "rag_agent", "artifact_agent", "tool_agent", "memory_agent", "skill_agent", "evaluator", "final_response"}
    merged: list[str] = []
    if isinstance(hints, list):
        for item in hints:
            if not isinstance(item, str):
                continue
            normalized = normalize_agent_name(item)
            if normalized in executable and normalized not in merged:
                merged.append(normalized)
    for item in rule_route:
        if item in executable and item not in merged:
            merged.append(item)
    if "evaluator" not in merged:
        merged.append("evaluator")
    if "final_response" not in merged:
        merged.append("final_response")
    return merged
