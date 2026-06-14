"""Planner Agent — deterministic rule-based routing for the multi-agent Supervisor.

No LLM dependency in the first version. Produces a RoutePlan that the
Supervisor graph uses for conditional routing.
"""

from typing import Any
import re

from src.web_app.agent.runtime.intent_schema import normalize_agent_name
from src.web_app.agent.runtime.state import AgentIntent, AgentRuntimeState, RiskLevel, RoutePlan
from src.web_app.agent.runtime.tool_routing import detect_local_tool, is_explicit_or_realtime_web_query


def _has_english_term(text: str, term: str) -> bool:
    """Word-boundary match for English keywords to avoid false positives
    like 'post' matching 'PostgreSQL' or 'send' matching 'send_async'."""
    return bool(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))

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

_BROAD_DOCUMENT_REFERENCE_TERMS = {
    "上传",
    "文档",
    "文件",
    "附件",
    "document",
    "file",
    "attachment",
}


def _has_term_match(
    text: str,
    terms: list[str],
    *,
    ignore_broad_document_terms: bool = False,
) -> bool:
    return any(
        term in text
        and not (ignore_broad_document_terms and term.lower() in _BROAD_DOCUMENT_REFERENCE_TERMS)
        for term in terms
    )

# Document overview / Q&A patterns — these are NOT research, always route to rag.
# Must be checked BEFORE _RESEARCH_TERMS to prevent misrouting.
_DOCUMENT_QA_KEYWORDS = [
    "文档里讲了啥", "文档里讲了什么", "文档讲了啥", "文档讲了什么",
    "这文档", "这个文件", "这份材料", "这篇报告",
    "这个文档", "这个附件", "附件里", "文件里",
    "总结一下这个", "总结一下文件", "概括一下", "概括这篇",
    "文档内容", "文件内容", "附件内容",
    "讲什么", "是什么内容", "主要说什么", "主要内容",
    "帮我看看这个文件", "帮我看看这个文档", "帮我读一下",
    "summarize this document", "what is this document", "what is this file about",
    "summarize the file",
]

_EXPLICIT_DOCUMENT_REFERENCE_KEYWORDS = [
    "根据文档", "根据文件", "根据材料", "根据附件", "根据我上传", "根据上传",
    "基于文档", "基于文件", "基于材料", "基于附件", "基于我上传", "基于上传",
    "上传的文档", "上传的文件", "上传的材料", "上传的附件",
    "我上传的文档", "我上传的文件", "我上传的材料", "我上传的附件",
    "当前文档", "当前文件", "这份文档", "这个文档", "这份文件", "这个文件",
    "这份材料", "这个材料", "附件里", "文档里", "文件里", "材料里",
    "this document", "this file", "the uploaded document", "the uploaded file",
    "uploaded document", "uploaded file", "attached document", "attached file",
    "in the attachment", "from the attachment",
    "based on the document", "based on the file", "according to the document",
    "according to the file", "summarize this document", "summarize this file",
    "what is this document", "what is this file",
    # Mojibake-compatible literals already present in historical tests/source.
    "鏍规嵁鏂囨。", "鏍规嵁鎴戜笂浼犵殑鏂囨。",
    "鎴戠殑鏂囨。", "杩欎釜鏂囦欢", "杩欎釜鏂囨。",
]


def _has_explicit_document_reference(text: str) -> bool:
    normalized = (text or "").lower()
    return any(
        keyword.lower() not in _BROAD_DOCUMENT_REFERENCE_TERMS and keyword in normalized
        for keyword in _EXPLICIT_DOCUMENT_REFERENCE_KEYWORDS
    )

_ARTIFACT_TERMS = [
    "生成文档", "方案", "网站", "页面", "代码", "图片提示词",
    "artifact", "生成一份", "写一份", "起草", "草稿", "文档",
    "产品方案", "MVP", "技术方案", "spec", "报告", "总结报告",
    "markdown", "网页", "prototype",
]

_TOOL_TERMS = [
    "发邮件", "发送邮件", "邮件", "评论", "操作", "打开", "点击",
    "提交", "发布", "删除", "修改外部系统", "付款", "支付", "转账",
    "执行", "运行命令",
    # ── Local file operations ──
    "创建文件", "写入文件", "修改文件", "保存到本地", "读取文件",
    "列出目录", "打开文件", "删除文件", "新建文件", "写文件",
    "写一个文件", "帮我创建", "帮我写", "写到本地", "写入本地",
    "保存文件", "本地文件", "读写文件",
    # ── Shell commands ──
    "运行命令", "执行脚本",
    "命令行", "终端", "执行命令", "跑命令",
    # ── Browser / form ──
    "打开网页", "填写表单",
    "发评论", "发布评论",
]

_EN_TOOL_KEYWORDS = [
    "email", "send", "post", "submit", "delete", "payment",
    "shell", "terminal", "cmd", "powershell",
]

_MEMORY_TERMS = [
    "记住", "以后", "我的偏好", "长期", "记忆", "保存下来",
    "下次记住", "别忘了", "记录一下",
    "remember", "save preference", "memory",
]

# ── Explicit memory-write patterns — these override research/rag/artifact ──
# When a user explicitly asks the agent to remember something, it is NOT a
# research task.  These patterns are checked with the highest priority.
_MEMORY_WRITE_PATTERNS = [
    "记住", "帮我记", "记一下", "记下来", "记录一下",
    "以后记得", "别忘了", "下次记住",
    "以后都", "以后要", "以后都要", "以后用", "以后给",
    "以后都是", "以后都要", "以后也得", "以后还想",
    "从此以后",
    "我的偏好是", "我的设置是",
    "我目标是", "我的目标是",
    "我的项目是", "我正在做",
    "默认用", "默认使用",
    "保存下来", "保存这个",
    "写入记忆", "存入记忆",
    "长期记忆", "永久记住",
    "remember", "save preference", "save my preference",
    "don't forget", "do not forget",
]

_STRONG_MEMORY_WRITE_PREFIXES = [
    "以后",  # "以后报告都要表格" — preference-setting
    "从此",  # "从此以后用中文" — long-term preference
    "从今",
]

# ── Explicit research request patterns ──────────────────────────────
# These override the memory/tech-stack guard and force is_research=True.
# Only trigger the full ODR pipeline when the user ASKS for deep research.

_RESEARCH_REQUEST_PATTERNS = [
    "帮我调研", "帮我研究", "做个研究", "做研究",
    "深度研究", "深度调研", "深度分析", "系统性调研",
    "调研", "研究报告", "研究报告", "最新研究",
    "deep research", "research report",
    "ODR", "odr",
]

# ── Tech stack / self-intro declarations → memory_confirm ──────────
# These are user statements ABOUT themselves/the project, not queries.
# Must be checked BEFORE _RAG_TERMS to prevent "Qdrant"/"PostgreSQL"
# from routing tech-stack declarations to RAG.
_TECH_STACK_DECLARATION_PATTERNS = [
    "这个项目用", "这个项目是", "项目技术栈", "项目用",
    "技术栈是", "技术栈：", "用的技术栈",
    "我的技术栈", "我的项目用", "我的项目是",
    "默认用", "默认使用",
    "我在用", "我用的", "我用的是",
    "我叫", "叫我", "我的名字", "我是",
]

# Name / identity preference patterns — "我叫C", "以后叫我C"
_NAME_PREFERENCE_PATTERNS = [
    "我叫", "叫我", "我的名字", "我是", "称呼我",
    "以后叫我", "称呼", "你可以叫我",
]

_SKILL_TERMS = [
    "复用", "下次复用", "工作流", "skill", "技能", "自动化流程",
    "沉淀", "做成模板", "标准化", "重复做", "自动化",
    "create skill", "reusable", "workflow", "流程",
]

# ── Conversation recall patterns ─────────────────────────────────
# These are questions about the CURRENT conversation history, not research/rag.
# They should route as chat + conversation_recall intent and never trigger
# "Evidence is insufficient" or external research agents.
_CONVERSATION_RECALL_PATTERNS = [
    "我问过你什么", "我刚才问过什么", "我之前问过什么",
    "我问过你", "是否问过", "有没有问过", "我问过.*相关",
    "刚才聊了什么", "我们刚才聊了什么", "上一条我问了什么",
    "刚刚问了什么", "之前问了什么", "上一轮",
    "之前聊过", "刚才说过", "我说过什么",
    "你还记得我问过", "你还记得我", "我提到过",
]

_CONVERSATION_RECALL_LITERALS = (
    "我前面问过", "我之前问过", "我刚才问过", "我刚刚问过",
    "前面问过", "之前问过", "刚才问过", "刚刚问过",
    "问过你啥", "问过你什么", "我问过你", "我前面问",
    "我之前问", "我刚才问", "会话记忆", "会话历史", "历史记录",
    "分析一下我的意图", "分析我的意图", "分析下我的意图",
)


def _looks_like_conversation_recall(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if any(literal in normalized for literal in _CONVERSATION_RECALL_LITERALS):
        return True
    current_context_terms = (
        "刚才", "刚刚", "前面", "之前", "上一条", "上一轮",
        "这个会话", "当前会话", "我们聊",
    )
    recall_verbs = ("记得", "记忆", "历史", "问过", "说过", "聊过")
    return any(term in normalized for term in current_context_terms) and any(
        verb in normalized for verb in recall_verbs
    )


def _is_project_diagnostic_question(user_input: str) -> bool:
    text = (user_input or "").lower()
    has_failure = any(term in text for term in (
        "\u5931\u8d25",  # 失败
        "\u62a5\u9519",  # 报错
        "\u9519\u8bef",  # 错误
        "\u6392\u67e5",  # 排查
        "\u770b\u54ea\u4e9b\u6a21\u5757",  # 看哪些模块
        "\u54ea\u4e9b\u6a21\u5757",  # 哪些模块
        "diagnostic", "troubleshoot", "failed", "error",
    ))
    has_project_area = any(term in text for term in (
        "\u4e0a\u4f20",  # 上传
        "\u6587\u6863",  # 文档
        "\u6587\u4ef6",  # 文件
        "\u5165\u5e93",  # 入库
        "\u6444\u5165",  # 摄入
        "embedding", "qdrant", "rag",
        "upload", "document", "ingest",
    ))
    return has_failure and has_project_area

# ── Public entry point ──────────────────────────────────────────────


def _infer_answer_mode(intent: str, user_input: str, *, is_memory_write: bool = False,
                        is_tech_stack: bool = False, is_name_pref: bool = False,
                        is_advice_question: bool = False,
                        is_project_diagnostic: bool = False,
                        is_conversation_recall: bool = False,
                        home_intent: dict[str, Any] | None = None) -> str:
    """Derive answer_mode from intent + input patterns + optional LLM hint.

    answer_mode controls:
    - How final_response phrases the answer (memory_confirm → "已记住")
    - Which memory categories ContextBuilder injects (MEMORY_CONTEXT_POLICY)
    """
    if is_conversation_recall:
        return "conversation_recall"
    if is_memory_write or is_name_pref:
        return "memory_confirm"
    if is_tech_stack and not is_advice_question:
        return "memory_confirm"
    if intent in ("research", "feed_research"):
        # LLM hint: if LLM says memory_confirm, trust it for project_advice override
        llm_am = str((home_intent or {}).get("answer_mode") or "")
        if llm_am == "memory_confirm":
            return "memory_confirm"
        return "project_advice"
    if intent == "rag" or str(intent).startswith("rag"):
        return "rag_qa"
    if str(intent).startswith("tool.") or str(intent).startswith("system.") or intent == "tool":
        return "tool_action"
    if intent == "artifact":
        return "project_advice"
    if intent == "chat":
        if is_project_diagnostic:
            return "project_advice"
        # Tech stack / memory-like declaration + advice question → project_advice
        if (is_tech_stack or is_name_pref) and is_advice_question:
            return "project_advice"
        # LLM hint: LLM may detect memory_confirm even when rules say chat
        llm_am = str((home_intent or {}).get("answer_mode") or "")
        if llm_am == "memory_confirm":
            return "memory_confirm"
        _casual_greetings = ("你好", "您好", "hi", "hello", "hey", "早", "晚安", "再见", "谢谢")
        if any(user_input.strip().lower().startswith(g) for g in _casual_greetings):
            return "casual"
        return "chat"
    if intent == "memory" or intent == "memory_confirm":
        return "memory_confirm"
    return "chat"


def plan_route(
    user_input: str,
    feed_card_id: int | str | None = None,
    matched_skills: list[dict[str, Any]] | None = None,
    context_summary: str = "",
    forced_route: str | None = None,
    forced_intent: str | None = None,
    home_intent: dict[str, Any] | None = None,
    has_document_attachments: bool = False,
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
    if forced in ("research", "rag", "artifact", "skill", "memory", "tool", "memory_write"):
        intent = forced  # type: ignore[assignment]
        reasons.append(f"forced_route={forced}")

    # ── Detect intent ───────────────────────────────────────────
    is_document_qa = has_document_attachments and _has_explicit_document_reference(text)
    is_research = any(term in text for term in _RESEARCH_TERMS) and not is_document_qa
    ignore_broad_document_terms = has_document_attachments and not is_document_qa
    is_rag = is_document_qa or _has_term_match(
        text,
        _RAG_TERMS,
        ignore_broad_document_terms=ignore_broad_document_terms,
    )
    is_artifact = _has_term_match(
        text,
        _ARTIFACT_TERMS,
        ignore_broad_document_terms=ignore_broad_document_terms,
    )
    is_tool = any(term in text for term in _TOOL_TERMS) or any(term in text for term in ["发邮件", "发送邮件", "邮件", "评论", "发布", "提交表单", "打开网页", "删除", "支付", "付款", "转账"]) or any(_has_english_term(text, kw) for kw in _EN_TOOL_KEYWORDS)
    is_memory = any(term in text for term in _MEMORY_TERMS)
    is_skill = any(term in text for term in _SKILL_TERMS)
    is_project_diagnostic = _is_project_diagnostic_question(user_input)
    local_tool = detect_local_tool(user_input)
    _has_explicit_research_request = any(
        pattern in text for pattern in _RESEARCH_REQUEST_PATTERNS
    )
    _has_light_web_search_request = (
        is_explicit_or_realtime_web_query(user_input)
        and not _has_explicit_research_request
        and not is_document_qa
        and not local_tool
    )
    if is_project_diagnostic and not forced:
        is_research = False
        is_rag = False
        is_artifact = False
        is_tool = False
        intent = "chat"
        reasons.append("project_diagnostic_graph_context")

    # ── Memory-guard: suppress tool detection for memory/context declarations ──
    _memory_prefixes = (
        "以后", "从此", "从今", "记住", "帮我记", "记一下",
        "这个项目用", "这个项目是", "项目技术栈", "项目用",
        "默认用", "默认使用", "不要再", "别再给我", "别再", "不要给我",
        "我偏好", "我的偏好", "我喜欢", "我习惯",
        "我不喜欢", "我不想要", "我讨厌",
        "我的项目", "我的技术栈", "我在用", "我用的",
        "我叫", "叫我", "我的名字", "我是",
        # ── English ──
        "i like", "i prefer", "i usually use", "i always use",
        "i don't like", "don't recommend", "remember that",
        "call me", "my name is",
    )
    _is_memory_like = any(user_input.strip().lower().startswith(p) for p in _memory_prefixes)
    # Tech stack declarations: "这个项目用X+Y+Z", "我的技术栈是..."
    _is_tech_stack = any(
        kw in text for kw in _TECH_STACK_DECLARATION_PATTERNS
    ) and not any(t in text for t in ("发邮件", "发送邮件", "执行命令", "删除"))
    # Name/identity preference: "我叫C", "以后叫我C"
    _is_name_preference = any(kw in text for kw in _NAME_PREFERENCE_PATTERNS)
    # Declaration + question: "这个项目用X怎么设计架构？"
    _has_advice_question = any(
        kw in text for kw in ("怎么", "如何", "设计架构", "架构设计", "方案建议", "帮我设计", "帮我看看")
    )
    _has_explicit_action = any(t in text for t in (
        "发邮件", "发送邮件", "创建文件", "写入文件", "删除文件",
        "执行命令", "运行命令", "发给", "发一封", "帮我创建",
    )) or any(_has_english_term(text, kw) for kw in ("email", "send", "delete"))
    if _is_memory_like and not _has_explicit_action:
        is_tool = False
        if _has_advice_question:
            # Declaration + question → project_advice (chat, no research)
            # e.g. "这个项目用X怎么设计架构" — user wants advice, not deep research
            is_research = False
            is_rag = False
            is_artifact = False
            reasons.append("memory_like_project_advice")
        else:
            is_rag = False
            is_research = False
            is_artifact = False
            is_memory = True
            reasons.append("memory_like_declaration")
    # ── Tech stack / name declarations → memory, NOT rag/research ──
    # Advice questions keep is_research=False; answer_mode=project_advice handles phrasing.
    if (_is_tech_stack or _is_name_preference) and not _has_explicit_action and not forced:
        if _has_advice_question:
            is_research = False
            is_rag = False
            is_artifact = False
            reasons.append("declaration_project_advice")
        else:
            is_rag = False
            is_research = False
            is_artifact = False
            is_memory = True
            reasons.append("declaration_to_memory")

    # ── Explicit research request overrides memory/tech-stack guard ──
    # Only when the user literally asks for deep research / investigation.
    if _has_explicit_research_request and not _has_explicit_action:
        is_research = True
        reasons.append("explicit_research_request")

    # ── Local read-only tools take priority over web search ────────
    if local_tool and not _has_explicit_action and not forced:
        is_tool = True
        is_research = False
        is_rag = False
        is_artifact = False
        intent = local_tool  # type: ignore[assignment]
        reasons.append(f"local_tool_detected({local_tool})")

    # ── Lightweight web search: separate from Open Deep Research ───
    if _has_light_web_search_request and not _has_explicit_action and not forced:
        is_tool = True
        is_research = False
        is_rag = False
        intent = "tool.web_search"
        reasons.append("light_web_search_request")

    # Conversation recall: "what did I just ask?" — must NOT trigger research/rag
    import re as _re
    is_conversation_recall = _looks_like_conversation_recall(text) or any(
        _re.search(pattern, text) for pattern in _CONVERSATION_RECALL_PATTERNS
    )
    # Explicit memory write request — highest priority, overrides research/rag/artifact
    is_memory_write = any(pattern in text for pattern in _MEMORY_WRITE_PATTERNS)

    llm_intent = str((home_intent or {}).get("intent") or (home_intent or {}).get("detected_intent") or "")
    _tool_intents = {"tool", "tool.email", "tool.local_file", "tool.web_search", "tool.browser", "tool.comment", "tool.form_submit", "tool.shell_readonly", "tool.shell_write", "tool.dangerous", "system.time", "system.calc", "system.unit_convert", "system.uuid", "system.hash"}
    # Declaration + advice question → project_advice; LLM must not override to research.
    _is_declaration_advice = (_is_tech_stack or _is_memory_like) and _has_advice_question and not _has_explicit_research_request
    _llm_override_blocked = (
        (_is_declaration_advice or is_project_diagnostic)
        and llm_intent in ("research", "feed_research", "mixed", "rag")
    ) or (
        (_has_light_web_search_request or bool(local_tool))
        and not _has_explicit_research_request
        and bool(llm_intent)
        and llm_intent not in ("tool", "tool.web_search", local_tool)
    )
    if not forced and llm_intent in {"chat", "research", "rag", "artifact", "feed_research", "memory", "skill", "mixed"} | _tool_intents and not _llm_override_blocked:
        intent = llm_intent  # type: ignore[assignment]
        reasons.append("home_intent_used")
        if llm_intent.startswith("tool.") or llm_intent in _tool_intents:
            is_tool = True
    elif _llm_override_blocked:
        reasons.append(f"llm_override_blocked(llm={llm_intent}→rule={intent})")

    # Feed card deep-dive takes priority
    if feed_card_id and is_research:
        intent = "feed_research"
        reasons.append("feed_card_deep_dive")
    elif feed_card_id:
        intent = "feed_research"
        reasons.append("feed_card_attached")

    # ── Explicit memory write takes priority over everything ──────
    # When the user says "记住" or "以后" + preference, it's a memory
    # write, not research — even if research/artifact keywords co-occur.
    # "以后都/要/用" patterns are preference-setting and override even tool.
    is_strong_memory_prefix = any(text.startswith(p) for p in _STRONG_MEMORY_WRITE_PREFIXES)
    if (is_memory_write or is_strong_memory_prefix):
        # When user is setting a preference ("以后都..."), it overrides
        # everything including tool detection.
        # When user says "记住...", only override research/rag/artifact,
        # not tool (e.g. "记住执行这个命令" should still be tool).
        if not is_tool or any(text.startswith(p) for p in _STRONG_MEMORY_WRITE_PREFIXES):
            intent = "memory"
            is_research = False
            is_rag = False
            is_artifact = False
            if any(text.startswith(p) for p in _STRONG_MEMORY_WRITE_PREFIXES):
                is_tool = False
            reasons.append("memory_write_priority")

    # ── Conversation recall: override research/rag/artifact ──────
    # "Did I ask about X?" is about current conversation history, NOT external research.
    if is_conversation_recall and not forced:
        intent = "chat"
        is_research = False
        is_rag = False
        is_artifact = False
        reasons.append("conversation_recall_detected")

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
    if intent == "tool" or str(intent).startswith("tool.") or str(intent).startswith("system.") or is_tool:
        route.append("tool_agent")
        reasons.append("tool_agent_in_route")
    # Document Q&A: force as rag, skip research/artifact/memory/skill
    if is_document_qa and not forced:
        intent = "document_qa"
        reasons = [
            reason
            for reason in reasons
            if reason
            not in {
                "mixed_keywords_detected",
                "research_agent_in_route",
                "rag_agent_in_route",
                "artifact_agent_in_route",
            }
        ]
        reasons.append("document_qa_detected")
        route = ["rag_agent"]  # only rag, no research/artifact
        reasons.append("rag_agent_in_route")

    # Post-execution: memory/skill BEFORE evaluator (so evaluator can score them)
    if intent in ("memory", "memory_confirm"):
        route.append("memory_agent")
    if is_skill or intent == "skill":
        route.append("skill_agent")

    # evaluator always before final_response
    route.append("evaluator")

    # final_response is always last
    route.append("final_response")

    # ── Risk assessment ─────────────────────────────────────────
    if is_tool:
        # ── Detect sub-intents for finer risk ───────────────────
        text = user_input.lower()
        _is_email = any(t in text for t in ["发邮件", "发送邮件", "邮件", "发一封"]) or _has_english_term(text, "email") or _has_english_term(text, "send")
        _is_local_write = any(t in text for t in ["创建文件", "写入文件", "修改文件", "保存到本地", "写文件", "写一个文件", "帮我创建", "帮我写", "写到本地", "写入本地", "保存文件", "新建文件", "写入"])
        _is_local_read = any(t in text for t in ["读取文件", "列出目录", "打开文件", "查看文件", "帮我看看", "看看本地", "列出文件", "查看目录", "看看文件", "列出"])
        _is_delete = any(t in text for t in ["删除", "remove", "rm "]) or _has_english_term(text, "delete")
        _is_shell = any(t in text for t in ["运行命令", "执行脚本", "shell", "terminal", "cmd", "powershell", "命令行", "终端", "执行命令", "跑命令"])
        _is_web_search = _has_light_web_search_request or intent == "tool.web_search"
        _is_browser = any(t in text for t in ["打开网页", "填写表单", "browser", "浏览器"])
        _is_form = any(t in text for t in ["提交", "发布评论", "发评论", "评论"]) or _has_english_term(text, "submit") or _has_english_term(text, "post")
        _is_local_system_tool = str(intent).startswith("system.")
        _is_high_risk = any(t in text for t in ["删除全部", "删除数据库", "支付", "付款", "转账", "删除项目", "删除所有", "全部删除",
                                                  "payment", "transfer", "drop database", "format", "shutdown",
                                                  "rm -rf", "sudo ", "chmod 777", "chown"])

        if _is_local_system_tool:
            risk_level = "L0"
            reasons.append("local_tool_l0")
        elif _is_high_risk:
            risk_level = "L4"
            needs_approval = True
            intent = "tool.dangerous"
            reasons.append("high_risk_l4_blocked")
        elif _is_web_search:
            risk_level = "L1"
            intent = "tool.web_search"
            reasons.append("web_search_l1")
        elif _is_delete:
            risk_level = "L4"
            needs_approval = True
            if _is_local_write:
                intent = "tool.local_file"
            reasons.append("delete_l4")
        elif _is_shell and (_is_local_write or _is_delete):
            risk_level = "L4"
            needs_approval = True
            intent = "tool.shell_write"
            reasons.append("shell_write_l4")
        elif _is_shell and not (_is_local_write or _is_delete):
            risk_level = "L2"
            intent = "tool.shell_readonly"
            reasons.append("shell_readonly_l2")
        elif _is_email and (any(t in text for t in ["发邮件", "发送邮件", "发一封"]) or _has_english_term(text, "send") or _has_english_term(text, "mail")):
            risk_level = "L3"
            needs_approval = True
            intent = "tool.email"
            reasons.append("email_send_l3")
        elif _is_local_write:
            risk_level = "L3"
            needs_approval = True
            intent = "tool.local_file"
            reasons.append("local_file_write_l3")
        elif _is_local_read:
            risk_level = "L1"
            intent = "tool.local_file" if not intent.startswith("tool.") else intent
            reasons.append("local_file_read_l1")
        elif _is_browser:
            risk_level = "L3"
            needs_approval = True
            intent = "tool.browser"
            reasons.append("browser_l3")
        elif _is_form:
            risk_level = "L3"
            needs_approval = True
            intent = "tool.form_submit"
            reasons.append("form_submit_l3")
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
        if not is_project_diagnostic:
            route = _merge_route(route, home_intent.get("suggested_route_hints") or home_intent.get("required_agents") or [])
        reasons.append("risk_guard_applied")

    # Expected output
    expected_output_map = {
        "research": "research_report",
        "feed_research": "research_report",
        "rag": "answer_with_evidence",
        "document_qa": "document_summary",
        "artifact": "document_or_code",
        "tool": "action_result",
        "tool.email": "action_result",
        "tool.local_file": "action_result",
        "tool.web_search": "answer_with_sources",
        "system.time": "local_tool_result",
        "system.calc": "local_tool_result",
        "system.unit_convert": "local_tool_result",
        "system.uuid": "local_tool_result",
        "system.hash": "local_tool_result",
        "tool.browser": "action_result",
        "tool.comment": "action_result",
        "tool.form_submit": "action_result",
        "tool.shell_readonly": "action_result",
        "tool.shell_write": "action_result",
        "tool.dangerous": "action_result",
        "skill": "skill_draft",
        "memory": "memory_update",
        "mixed": "structured_report",
        "chat": "answer",
    }

    answer_mode = _infer_answer_mode(
        intent, user_input,
        is_memory_write=is_memory_write,
        is_tech_stack=_is_tech_stack,
        is_name_pref=_is_name_preference,
        is_advice_question=_has_advice_question or is_project_diagnostic,
        is_project_diagnostic=is_project_diagnostic,
        is_conversation_recall=is_conversation_recall,
        home_intent=home_intent,
    )
    if answer_mode == "conversation_recall":
        route = [item for item in route if item not in {"memory_agent", "rag_agent", "research_agent", "artifact_agent", "tool_agent"}]
        if "evaluator" not in route:
            route.append("evaluator")
        if "final_response" not in route:
            route.append("final_response")
    route_plan: RoutePlan = {
        "intent": intent,
        "route": route,
        "risk_level": risk_level,
        "needs_approval": needs_approval,
        "expected_output": expected_output_map.get(intent, "answer"),
        "reason": "; ".join(reasons) if reasons else "default_chat_route",
        "answer_mode": answer_mode,
        "explicit_research": bool(_has_explicit_research_request),
        "research_mode": "deep" if _has_explicit_research_request and "research_agent" in route else ("fallback" if "research_agent" in route else "none"),
        "memory_context_loader": answer_mode != "conversation_recall",
        "memory_writer_planned": "memory_agent" in route,
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
