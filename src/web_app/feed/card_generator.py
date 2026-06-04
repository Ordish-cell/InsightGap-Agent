import re
from typing import Any

_DOMAIN_CN = {
    "agent": "Agent",
    "rag": "RAG",
    "devtools": "开发工具",
    "startup": "创业机会",
    "research": "研究前沿",
    "ai": "AI",
}

_INTEREST_CN_MAP = {
    "agent": "Agent 技术",
    "rag": "RAG 检索增强",
    "mcp": "MCP 协议",
    "langgraph": "LangGraph 工作流",
    "langchain": "LangChain 框架",
    "skill": "Skill 复用",
    "memory": "Memory 记忆",
    "feed": "Feed 推荐",
    "deep research": "Deep Research",
    "devtools": "开发工具链",
    "research": "AI 研究",
    "startup": "产品机会",
}

_BENEFIT_TEMPLATES = {
    "agent": [
        "可以帮助你理解 Agent 架构的最新演进方向，优化自己的 Agent OS 设计。",
        "能为你的 Agent 运行时增加一种新的上下文组织或工具调用思路。",
        "对 Agent 的自主决策和质量评估有直接参考价值。",
    ],
    "rag": [
        "可以帮助你改进 RAG 检索精度和上下文拼接策略。",
        "能启发你在 Qdrant + LangChain 方案上做更精准的 chunk 设计和召回。",
        "对知识库产品的检索体验提升有直接帮助。",
    ],
    "devtools": [
        "可以帮助你优化开发工作流或选择更合适的技术组件。",
        "能减少重复造轮子，直接复用社区已验证的工程方案。",
        "对 FastAPI + Vite + React 技术栈的稳定性或性能有提升。",
    ],
    "startup": [
        "可以帮助你发现信息差 Agent OS 的产品差异化机会。",
        "能让你更早判断某个方向的竞争格局和入场时机。",
        "对产品定位和功能优先级排序有参考价值。",
    ],
    "research": [
        "可以帮助你判断某个技术路线是否值得深入研究或集成。",
        "能让你在技术选型时少走弯路，直接参考最新基准结果。",
        "对 Agent OS 的底层能力（推理、检索、评估）有提升潜力。",
    ],
    "ai": [
        "可以帮助你把握 AI 领域的整体趋势，发现跨领域的产品灵感。",
        "能让你在信息差 Agent OS 的规划中保持技术敏感度。",
        "对理解用户需求和市场方向有帮助。",
    ],
}

_GAP_TEMPLATES = {
    "agent": [
        "多数人只把它当一篇 Agent 论文看，但它提出的方法可以直接影响 Agent OS 的 Skill 自动生成和质量评估体系。",
        "表面上是在讨论 Agent 评估，实际上它揭示了一种可复用的能力沉淀模式，这正是你的 Skill 系统需要的。",
        "看起来是学术研究，但它的核心思路可以反向指导 Agent 运行时的上下文组织和工具选择策略。",
    ],
    "rag": [
        "多数人只关注检索速度，但它的核心贡献在于上下文质量——这直接影响 Agent 回答的准确性。",
        "表面上是一篇检索论文，但它的分块和重排序策略可以迁移到你的知识库产品中。",
        "大部分人忽略了它对多源异构文档的处理方式，而这恰好是 Agent OS 知识库的痛点。",
    ],
    "devtools": [
        "多数人把它当普通开源项目看，但它解决的工作流自动化问题正是 Agent OS 开发效率的关键。",
        "表面上是个工具，但它背后的设计模式可以复用到你的 Agent 工具链中。",
        "大部分人只关注功能列表，忽略了它的架构决策对类似系统的参考价值。",
    ],
    "startup": [
        "多数人把它当行业新闻消费，但它背后反映的用户需求变化可能直接影响你的产品方向。",
        "表面上是一个融资或产品发布消息，但它验证的市场需求和你正在做的信息差 Agent OS 高度相关。",
        "大部分人会忽略它对竞争格局的暗示，但早期信号往往藏在这样的信息里。",
    ],
    "research": [
        "多数人只把它当一篇论文存档，但它提出的方法和你的 Deep Research 能力直接相关。",
        "表面上在讨论基准测试，但它揭示的能力瓶颈恰好是你下一步要优化的方向。",
        "大部分人关注结果数字，但它的实验设计和消融研究对工程落地更有启发。",
    ],
    "ai": [
        "多数人只会扫一眼标题，但它背后的技术趋势可能在未来几个月影响你的产品决策。",
        "表面上是一条普通的 AI 动态，但它连接了多个你关注的技术领域。",
        "大部分人看过就忘，但如果你把它和 Agent OS 的路线图对照，会发现有价值的技术信号。",
    ],
}

_NEXT_ACTIONS = [
    "带入对话，让 Agent 分析这条信息对你当前阶段的具体启发。",
    "深度研究，生成一份包含落地建议和风险提示的完整报告。",
    "保存到知识库，作为后续 Skill 设计或技术决策的参考素材。",
    "让 Agent 提炼可复用的方法论，沉淀为 Skill。",
]

_VALUE_PREFIXES = {
    "agent": ["这条信息揭示了 Agent 系统设计的一个新思路：", "它提示你 Agent 能力可以这样扩展：", "这项研究为 Agent 产品化提供了一个新角度："],
    "rag": ["这条信息展示了 RAG 技术的一个改进方向：", "它提示你检索增强可以从这个维度优化：", "这项研究为知识库产品提供了一个新思路："],
    "devtools": ["这个项目展示了一种更高效的开发方式：", "它提供了一个可以集成到 Agent OS 的工具思路：", "这个工具解决了一个开发效率痛点："],
    "startup": ["这条信息暗示了一个产品机会：", "它验证了市场对某类 AI 产品的需求：", "这个动态可能影响你的产品优先级："],
    "research": ["这项研究的结论值得关注：", "它揭示了一个可能改变技术路线的新发现：", "这个研究方向可能成为下一个能力突破点："],
    "ai": ["这条信息标记了一个值得关注的趋势：", "它连接了多个你关注的技术方向：", "这个动态可能影响 AI 产品的下一阶段演进："],
}

_WHY_RELEVANT_PREFIXES = {
    "agent": ["与你正在构建的 Agent OS 直接相关，", "你的 Agent 运行时架构可以从这项工作中借鉴思路，", "你当前阶段的 Agent 能力建设正好需要这类方案，"],
    "rag": ["你的知识库产品依赖 RAG 能力，", "你的 Agent 检索和上下文构建正好需要这类优化，", "你基于 Qdrant 的 RAG 方案可以从中获得改进灵感，"],
    "devtools": ["你的 FastAPI + Vite + React 技术栈可以从这个工具中受益，", "这个项目解决的工作流问题正是你开发效率的关键，", "它可以减少你在工具链上的试错成本，"],
    "startup": ["你正在做的信息差 Agent OS 属于这个赛道，", "这条信息能帮你判断产品定位是否准确，", "它可能影响你下一步的功能优先级决策，"],
    "research": ["你的 Deep Research 能力可以从这项研究中获得方法升级，", "它和你的 Agent OS 研究模块高度相关，", "这项研究的结论可能影响你的技术路线选择，"],
    "ai": ["它和你关注的多个技术方向都有交集，", "这条信息能帮你保持对 AI 趋势的敏感度，", "它可能启发你的产品规划或技术选型，"],
}


def generate_feed_card(info_item: Any, score: dict[str, Any], user_profile: Any) -> dict[str, Any]:
    domain = (info_item.raw_metadata or {}).get("domain", "ai")
    tags = (info_item.raw_metadata or {}).get("tags", info_item.topics or [])
    source_type = info_item.source_type or "web"
    interests = getattr(user_profile, "explicit_interests", None) or ["Agent", "RAG"]
    original_title = info_item.title

    chinese_title = _generate_chinese_title(original_title, source_type, tags, domain)
    one_sentence_value = _generate_one_sentence_value(original_title, domain, tags, source_type)
    why_relevant = _generate_why_relevant(original_title, domain, tags, interests, source_type)
    benefit = _generate_benefit(domain, tags)
    information_gap = _generate_information_gap(domain, tags, source_type)
    next_action = _generate_next_action(domain, source_type)
    summary = info_item.summary if info_item.summary and _contains_chinese(info_item.summary) else _generate_chinese_summary(original_title, info_item.summary, domain)

    evidence = [
        {
            "title": original_title,
            "url": info_item.source_url or None,
            "source_type": source_type,
            "credibility": score["source_credibility"],
            "published_at": info_item.published_at.isoformat() if info_item.published_at else None,
            "snippet": (info_item.summary or "")[:300],
        }
    ]

    return {
        "card_type": "insight",
        "title": chinese_title,
        "one_sentence_value": one_sentence_value,
        "why_you": why_relevant,
        "information_gap": information_gap,
        "summary": summary,
        "source_type": source_type,
        "domain": domain,
        "relation_type": score["relation_type"],
        "evidence": evidence,
        "suggested_actions": ["use_in_chat", "deep_research", "view_detail", "save", "ignore"],
        "score": score,
        "final_score": score["final"],
        "confidence": score["confidence"],
        "status": "active",
        "why_relevant": why_relevant,
        "benefit": benefit,
        "next_action": next_action,
        "original_title": original_title,
    }


# Mapping of English technical terms to Chinese for title generation.
# Proper nouns (RAG, MCP, LangGraph, Agent, Skill, LLM, etc.) are kept as-is.
_TERM_CN = {
    "agent": "Agent", "skill": "Skill", "rag": "RAG", "mcp": "MCP",
    "memory": "记忆", "evaluation": "评估", "reasoning": "推理",
    "chain-of-thought": "思维链", "cot": "思维链", "retrieval": "检索",
    "llm": "LLM", "graph": "图谱", "workflow": "工作流",
    "multi-agent": "多Agent", "benchmark": "基准", "steering": "引导",
    "controllable": "可控", "efficient": "高效", "unifying": "统一",
    "heterogeneous": "异构", "criteria": "标准", "knowledge": "知识",
    "generation": "生成", "augmented": "增强", "learning": "学习",
    "training": "训练", "fine-tuning": "微调", "embedding": "嵌入",
    "planning": "规划", "tool": "工具", "automation": "自动化",
    "orchestration": "编排", "deployment": "部署", "scaling": "扩展",
    "search": "搜索", "browser": "浏览器", "web": "Web",
    "ui": "UI", "interface": "界面", "api": "API", "database": "数据库",
    "langgraph": "LangGraph", "langchain": "LangChain",
    "open-source": "开源", "framework": "框架", "system": "系统",
    "model": "模型", "design": "设计", "architecture": "架构",
    "security": "安全", "privacy": "隐私", "optimization": "优化",
    "inference": "推理", "deploy": "部署", "monitoring": "监控",
    "collaboration": "协作", "synthesis": "合成", "alignment": "对齐",
    "safety": "安全", "robustness": "鲁棒", "scalability": "可扩展",
    "supervised": "监督", "unsupervised": "无监督", "adaptive": "自适应",
    "dynamic": "动态", "static": "静态", "hybrid": "混合", "unified": "统一",
    "distributed": "分布式", "federated": "联邦", "incremental": "增量",
    "online": "在线", "real-time": "实时", "streaming": "流式",
    "multimodal": "多模态", "cross-modal": "跨模态", "vision": "视觉",
    "language": "语言", "speech": "语音", "audio": "音频", "video": "视频",
    "code": "代码", "text": "文本", "document": "文档", "image": "图像",
}


def _build_cn_topic(title: str, tags: list[str], domain: str) -> str:
    """Build a fully Chinese topic phrase from title keywords and tags."""
    # Priority 1: use meaningful tags mapped to Chinese
    meaningful = [t for t in tags if t.lower() not in
                  ("paper", "arxiv", "cs.lg", "cs.cl", "cs.ai", "cs.cv", "github", "rss", "blog")]
    cn_parts = []
    seen = set()
    for tag in meaningful[:4]:
        cn = _TERM_CN.get(tag.lower(), "")
        if cn and cn not in seen:
            cn_parts.append(cn)
            seen.add(cn)

    # Priority 2: extract significant words from title and translate
    if len(cn_parts) < 3:
        words = re.findall(r"[A-Z][a-z]+|[A-Z]{2,}|[a-z]{4,}", title)
        stop = {"paper", "arxiv", "github", "using", "based", "from", "with",
                "new", "large", "model", "method", "learning", "deep", "research",
                "study", "approach", "towards", "through", "novel", "improved",
                "efficient", "controllable", "heterogeneous"}
        for w in words:
            if w.lower() in stop:
                continue
            cn = _TERM_CN.get(w.lower(), "")
            if cn and cn not in seen:
                cn_parts.append(cn)
                seen.add(cn)
            elif w.lower() not in stop and w not in seen:
                # Keep proper nouns / acronyms as-is
                if w.isupper() and len(w) <= 6:
                    cn_parts.append(w)
                    seen.add(w)

    # Priority 3: use domain as fallback
    if not cn_parts:
        domain_cn = _DOMAIN_CN.get(domain, "AI")
        return f"{domain_cn}技术"

    topic = " ".join(cn_parts[:4])
    # Add domain context if topic is short (1-2 words)
    if len(cn_parts) <= 2:
        domain_cn = _DOMAIN_CN.get(domain, "AI")
        return f"{domain_cn}{topic}"
    return topic


def _generate_chinese_title(original_title: str, source_type: str, tags: list[str], domain: str) -> str:
    clean = " ".join((original_title or "").split())
    if not clean:
        return "未命名信息差"

    if _contains_chinese(clean) and len(clean) <= 36:
        return clean
    if _contains_chinese(clean):
        return clean[:33] + "..."

    if ":" in clean and source_type in ("arxiv", "paper"):
        return _arxiv_title(clean, domain, tags)

    if source_type == "github":
        return _github_title(clean, domain, tags)

    return _generic_cn_title(clean, domain, tags, source_type)


def _arxiv_title(title: str, domain: str, tags: list[str]) -> str:
    parts = title.split(":", 1)
    main = parts[1].strip() if len(parts) > 1 else parts[0].strip()
    main = main.rstrip(".")
    topic = _build_cn_topic(main, tags, domain)
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    templates = [
        f"关于{topic}的新研究",
        f"{topic}：{domain_cn}领域新进展",
        f"用{topic}改进{domain_cn}的方法",
        f"一项关于{topic}的{domain_cn}研究",
        f"{topic}方向的技术新思路",
        f"从新论文看{topic}的演进趋势",
    ]
    idx = hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 36:
        result = result[:33] + "..."
    return result


def _github_title(title: str, domain: str, tags: list[str]) -> str:
    topic = _build_cn_topic(title, tags, domain)
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    templates = [
        f"值得关注的{topic}开源项目",
        f"{topic}方向的实用开源工具",
        f"一个{topic}相关的{domain_cn}开源项目",
        f"GitHub 上的{topic}新项目",
        f"适合{domain_cn}场景的{topic}开源方案",
    ]
    idx = hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 36:
        result = result[:33] + "..."
    return result


def _generic_cn_title(title: str, domain: str, tags: list[str], source_type: str) -> str:
    topic = _build_cn_topic(title, tags, domain)
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    templates = [
        f"{topic}：值得关注的{domain_cn}信号",
        f"关于{topic}的新信息差",
        f"从{topic}看{domain_cn}的新变化",
        f"{domain_cn}方向的新发现：{topic}",
        f"{topic}领域的最新动态",
        f"一条关于{topic}的高价值信息",
    ]
    idx = hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 36:
        result = result[:33] + "..."
    return result


def _generate_one_sentence_value(title: str, domain: str, tags: list[str], source_type: str) -> str:
    prefixes = _VALUE_PREFIXES.get(domain, _VALUE_PREFIXES["ai"])
    idx = hash(title + "value") % len(prefixes)
    keywords = _extract_keywords(title, tags)
    suffix_templates = [
        f"它可能改变你对{keywords}的理解和技术选型。",
        f"它提供了一种可复用的思路，能直接改进你的{_DOMAIN_CN.get(domain, 'AI')}模块。",
        f"它标记了一个容易被忽略但实际很重要的技术信号。",
        f"它把{keywords}和你当前的产品方向连接了起来。",
    ]
    suffix_idx = hash(title + "suffix") % len(suffix_templates)
    return prefixes[idx] + suffix_templates[suffix_idx]


def _generate_why_relevant(title: str, domain: str, tags: list[str], interests: list[str], source_type: str) -> str:
    prefixes = _WHY_RELEVANT_PREFIXES.get(domain, _WHY_RELEVANT_PREFIXES["ai"])
    idx = hash(title + "why") % len(prefixes)
    matched = [t for t in tags if t.lower() in " ".join(interests).lower()]
    if matched:
        suffix = f"尤其涉及你关注的 {matched[0]}。"
    else:
        suffix = f"和你的信息差 Agent OS 技术路线有交集。"
    return prefixes[idx] + suffix


def _generate_benefit(domain: str, tags: list[str]) -> str:
    templates = _BENEFIT_TEMPLATES.get(domain, _BENEFIT_TEMPLATES["ai"])
    idx = hash(str(tags)) % len(templates)
    return templates[idx]


def _generate_information_gap(domain: str, tags: list[str], source_type: str) -> str:
    templates = _GAP_TEMPLATES.get(domain, _GAP_TEMPLATES["ai"])
    idx = hash(str(tags) + "gap") % len(templates)
    return templates[idx]


def _generate_next_action(domain: str, source_type: str) -> str:
    idx = hash(domain + source_type) % len(_NEXT_ACTIONS)
    return _NEXT_ACTIONS[idx]


def _generate_chinese_summary(original_title: str, summary: str, domain: str) -> str:
    if summary and len(summary) > 20:
        return summary[:200]
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    return f"这条来自{domain_cn}领域的信息值得进一步研究，可能对你的产品和技术决策有参考价值。"


def _extract_keywords(text: str, tags: list[str]) -> str:
    """Extract key technical terms and return as Chinese phrase."""
    # First try to use tags mapped to Chinese
    meaningful = [t for t in tags if t.lower() not in
                  ("paper", "arxiv", "cs.lg", "cs.cl", "cs.ai", "cs.cv", "github", "rss", "blog")]
    cn_parts = []
    for tag in meaningful[:3]:
        cn = _TERM_CN.get(tag.lower(), "")
        if cn:
            cn_parts.append(cn)
        elif tag.isupper() and len(tag) <= 6:
            cn_parts.append(tag)
    if cn_parts:
        return " ".join(cn_parts)

    # Fallback: extract from text
    words = re.findall(r"[A-Z][a-z]+|[A-Z]{2,}|[a-z]{4,}", text)
    stop = {"paper", "arxiv", "github", "using", "based", "from", "with",
            "new", "large", "model", "method", "learning", "deep", "research",
            "study", "approach", "towards", "through", "novel", "improved"}
    result = []
    for w in words:
        if w.lower() in stop:
            continue
        cn = _TERM_CN.get(w.lower(), "")
        if cn:
            result.append(cn)
        elif w.isupper() and len(w) <= 6:
            result.append(w)
    if result:
        return " ".join(result[:3])
    return "AI 技术"


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def is_mostly_english(text: str) -> bool:
    """Check if text is predominantly English (not Chinese)."""
    if not text:
        return False
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    # If there are very few CJK characters and lots of ASCII letters, it's mostly English
    return ascii_letters > cjk * 2 and cjk < 3


def generate_display_title(title: str, source_type: str = "web", tags: list[str] | None = None, domain: str = "ai") -> str:
    """Public entry point: ensure a display title is Chinese.
    Used by card_to_dict for old DB cards that still have English titles.
    """
    if not title:
        return "未命名信息差"
    if not is_mostly_english(title):
        return title
    return _generate_chinese_title(title, source_type, tags or [], domain)
