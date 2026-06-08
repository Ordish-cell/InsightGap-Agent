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
        "你可以用「{title}」的思路检查当前 Agent OS 的 Skill 自动生成和质量评估模块是否需要调整。",
        "「{title}」能为你的 Agent 运行时增加一种新的上下文组织或工具调用思路。",
        "「{title}」对 Agent 的自主决策和质量评估有直接参考价值。",
        "从这个工作出发，你可以检查当前 Memory 和 Skill 系统是否需要适配。",
        "它提供的方法可以迁移到你的 Agent OS（{domain_cn}场景），可能替换或增强现有模块。",
        "它的设计决策可以帮助你避免在类似问题上踩坑，加速 Agent OS 迭代。",
    ],
    "rag": [
        "你可以用「{title}」改进 RAG 检索精度和上下文拼接策略。",
        "「{title}」能启发你在 Qdrant + LangChain 方案上做更精准的 chunk 设计和召回。",
        "「{title}」对知识库产品的检索体验提升有直接帮助。",
        "你可以对照这条信息，检查当前 RAG pipeline 的 chunk→embed→retrieve 三阶段。",
        "它的方案可能直接替换你现有检索链路中的某个薄弱环节。",
    ],
    "devtools": [
        "「{title}」可以帮助你优化开发工作流或选择更合适的技术组件。",
        "「{title}」能减少重复造轮子，直接复用社区已验证的工程方案。",
        "「{title}」对 FastAPI + Vite + React 技术栈的稳定性或性能有提升。",
        "你可以评估是否将其纳入 Agent OS 的工具链，替代当前自研方案。",
    ],
    "startup": [
        "「{title}」可以帮助你发现信息差 Agent OS 的产品差异化机会。",
        "「{title}」能让你更早判断某个方向的竞争格局和入场时机。",
        "「{title}」对产品定位和功能优先级排序有参考价值。",
        "你可以把这条信号纳入产品路线图讨论，判断是否影响下一阶段优先级。",
    ],
    "research": [
        "「{title}」可以帮助你判断某个技术路线是否值得深入研究或集成。",
        "「{title}」能让你在技术选型时少走弯路，直接参考最新基准结果。",
        "「{title}」对 Agent OS 的底层能力（推理、检索、评估）有提升潜力。",
        "你可以把它的结论作为 Deep Research 模块的输入，生成一篇专题研究报告。",
    ],
    "ai": [
        "「{title}」可以帮助你把握 AI 领域的整体趋势，发现跨领域的产品灵感。",
        "「{title}」能让你在信息差 Agent OS 的规划中保持技术敏感度。",
        "「{title}」对理解用户需求和市场方向有帮助。",
    ],
    # far_domain 专属 benefit —— 不写 Agent 技术
    "far_domain": [
        "「{title}」能帮你理解非技术领域的信号捕捉和反馈回路设计，这是 Feed 远域启发模块的核心能力。",
        "「{title}」可以用于设计远域启发卡的评分与解释规则，把市场/用户信号转成可行动判断。",
        "「{title}」在信号筛选和机会发现上的思路，可以迁移到 Feed 的非同温层信息采集流程中。",
        "「{title}」展示的反馈回路机制，可以帮助你把信息差从技术资讯扩展为机会发现系统。",
    ],
}

_GAP_TEMPLATES = {
    "agent": [
        "多数人只把「{title}」当一篇 Agent 论文看，但它提出的方法可以直接影响 Agent OS 的 Skill 自动生成和质量评估体系。",
        "表面上「{title}」在讨论 Agent 评估，实际上它揭示了一种可复用的能力沉淀模式，这正是你的 Skill 系统需要的。",
        "「{title}」看起来是学术研究，但它的核心思路可以反向指导 Agent 运行时的上下文组织和工具选择策略。",
        "大部分人忽略了「{title}」中对自主决策链路的设计，而这恰好是 Agent OS 区别于普通 Chatbot 的关键。",
    ],
    "rag": [
        "多数人只关注「{title}」里的检索速度，但它的核心贡献在于上下文质量——这直接影响 Agent 回答的准确性。",
        "表面上「{title}」是一篇检索论文，但它的分块和重排序策略可以迁移到你的知识库产品中。",
        "大部分人忽略了「{title}」对多源异构文档的处理方式，而这恰好是 Agent OS 知识库的痛点。",
    ],
    "devtools": [
        "多数人把「{title}」当普通开源项目看，但它解决的工作流自动化问题正是 Agent OS 开发效率的关键。",
        "表面上「{title}」是个工具，但它背后的设计模式可以复用到你的 Agent 工具链中。",
        "大部分人只关注「{title}」的功能列表，忽略了它的架构决策对类似系统的参考价值。",
    ],
    "startup": [
        "多数人把「{title}」当行业新闻消费，但它背后反映的用户需求变化可能直接影响你的产品方向。",
        "表面上「{title}」是一个融资或产品发布消息，但它验证的市场需求和你正在做的信息差 Agent OS 高度相关。",
        "大部分人会忽略「{title}」对竞争格局的暗示，但早期信号往往藏在这样的信息里。",
    ],
    "research": [
        "多数人只把「{title}」当一篇论文存档，但它提出的方法和你的 Deep Research 能力直接相关。",
        "表面上「{title}」在讨论基准测试，但它揭示的能力瓶颈恰好是你下一步要优化的方向。",
        "大部分人关注「{title}」的结果数字，但它的实验设计和消融研究对工程落地更有启发。",
    ],
    "ai": [
        "多数人只会扫一眼「{title}」的标题，但它背后的技术趋势可能在未来几个月影响你的产品决策。",
        "表面上「{title}」是一条普通的 AI 动态，但它连接了多个你关注的技术领域。",
        "大部分人看过「{title}」就忘，但如果你把它和 Agent OS 的路线图对照，会发现有价值的技术信号。",
    ],
    # far_domain 专属 —— 不写 Agent 直接相关
    "far_domain": [
        "多数人把「{title}」当一条普通行业信息消费，但它展示的信号筛选和机会发现模式可以直接启发 Feed 远域模块的设计。",
        "表面上「{title}」和你做的 Agent 系统没有直接关系，但它的反馈回路和信号捕捉机制值得远域启发模块借鉴。",
        "这条信息本身不属于 Agent/RAG 圈，但展示了弱信号如何被产品化捕捉和放大。",
        "远域启发点不在技术栈，而在信号筛选、反馈回路和机会发现流程——这正是「{title}」的价值所在。",
    ],
}

_NEXT_ACTIONS = [
    "把「{title}」带入对话，让 Agent 分析这条信息对你当前阶段的具体启发。",
    "对「{title}」做一次深度研究，输出可迁移到 Agent OS 的 3 个设计点。",
    "保存「{title}」到知识库，作为后续 Skill 设计或技术决策的参考素材。",
    "让 Agent 从「{title}」中提炼可复用的方法论，沉淀为 Skill。",
    "对照「{title}」检查当前 Agent OS 的相关模块是否有改进空间。",
    "将「{title}」的核心发现记录到记忆系统，供后续对话自动引用。",
]

_VALUE_PREFIXES = {
    "agent": ["这条信息揭示了 Agent 系统设计的一个新思路：", "它提示你 Agent 能力可以这样扩展：", "这项研究为 Agent 产品化提供了一个新角度：", "它展示了一种可以迁移到 Agent OS 的方法："],
    "rag": ["这条信息展示了 RAG 技术的一个改进方向：", "它提示你检索增强可以从这个维度优化：", "这项研究为知识库产品提供了一个新思路：", "它提供了一种提升检索质量的方法："],
    "devtools": ["这个项目展示了一种更高效的开发方式：", "它提供了一个可以集成到 Agent OS 的工具思路：", "这个工具解决了一个开发效率痛点：", "它可能替代你当前工具链中的某个环节："],
    "startup": ["这条信息暗示了一个产品机会：", "它验证了市场对某类 AI 产品的需求：", "这个动态可能影响你的产品优先级：", "它揭示了一个值得关注的竞争信号："],
    "research": ["这项研究的结论值得关注：", "它揭示了一个可能改变技术路线的新发现：", "这个研究方向可能成为下一个能力突破点：", "它的实验结论对工程落地有直接启发："],
    "ai": ["这条信息标记了一个值得关注的趋势：", "它连接了多个你关注的技术方向：", "这个动态可能影响 AI 产品的下一阶段演进：", "它提供了一个跨领域的灵感信号："],
    # far_domain 专属 —— 非 Agent 语言
    "far_domain": [
        "这条来自远域的信息展示了一种信号捕捉模式：",
        "它来自非技术领域，但反馈机制值得借鉴：",
        "这条信息标记了一个容易被技术圈忽略的机会信号：",
        "它展示了如何把市场变化转成可行动的判断依据：",
    ],
}

_WHY_RELEVANT_PREFIXES = {
    "agent": ["与你正在构建的 Agent OS 直接相关，", "你的 Agent 运行时架构可以从「{title}」中借鉴思路，", "你当前阶段的 Agent 能力建设正好需要这类方案，"],
    "rag": ["你的知识库产品依赖 RAG 能力，", "你的 Agent 检索和上下文构建正好需要「{title}」里的优化，", "你基于 Qdrant 的 RAG 方案可以从「{title}」中获得改进灵感，"],
    "devtools": ["你的 FastAPI + Vite + React 技术栈可以从「{title}」中受益，", "「{title}」解决的工作流问题正是你开发效率的关键，", "「{title}」可以减少你在工具链上的试错成本，"],
    "startup": ["你正在做的信息差 Agent OS 属于这个赛道，", "「{title}」能帮你判断产品定位是否准确，", "「{title}」可能影响你下一步的功能优先级决策，"],
    "research": ["你的 Deep Research 能力可以从「{title}」中获得方法升级，", "「{title}」和你的 Agent OS 研究模块高度相关，", "「{title}」的结论可能影响你的技术路线选择，"],
    "ai": ["「{title}」和你关注的多个技术方向都有交集，", "「{title}」能帮你保持对 AI 趋势的敏感度，", "「{title}」可能启发你的产品规划或技术选型，"],
    # far_domain 专属 —— 不以 Agent 技术为核心
    "far_domain": [
        "它不是你当前技术栈的直接内容，但能启发 Feed 如何发现非同温层机会，",
        "它能帮助你把信息差从技术资讯扩展为机会发现系统，",
        "这条信息来自远域，价值不在于技术栈匹配，而在于信号捕捉和反馈回路设计，",
    ],
}


def generate_feed_card(info_item: Any, score: dict[str, Any], user_profile: Any) -> dict[str, Any]:
    domain = (info_item.raw_metadata or {}).get("domain", "ai")
    tags = (info_item.raw_metadata or {}).get("tags", info_item.topics or [])
    source_type = info_item.source_type or "web"
    interests = getattr(user_profile, "explicit_interests", None) or ["Agent", "RAG"]
    original_title = info_item.title
    relation_type = score.get("relation_type", "far_domain")
    source_kind = (info_item.raw_metadata or {}).get("source_kind", "")

    chinese_title = _generate_chinese_title(original_title, source_type, tags, domain, relation_type, source_kind)
    one_sentence_value = _generate_one_sentence_value(original_title, domain, tags, source_type, relation_type)
    why_relevant = _generate_why_relevant(original_title, domain, tags, interests, source_type, relation_type)
    benefit = _generate_benefit(domain, tags, original_title, relation_type)
    information_gap = _generate_information_gap(domain, tags, source_type, original_title, relation_type)
    next_action = _generate_next_action(domain, source_type, original_title)
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

    card = {
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

    card = validate_card_quality(card)
    return card


def validate_card_quality(card: dict[str, Any]) -> dict[str, Any]:
    """Ensure card fields are non-empty and non-generic. Degrade to plain format if invalid."""
    title = str(card.get("title", ""))
    info_gap = str(card.get("information_gap", ""))
    why_relevant = str(card.get("why_relevant", ""))
    benefit = str(card.get("benefit", ""))
    next_action = str(card.get("next_action", ""))
    evidence = card.get("evidence", [])
    original_title = str(card.get("original_title", ""))

    issues = []
    if not title or title == "未命名信息差":
        issues.append("title_empty")
    if not info_gap:
        issues.append("information_gap_empty")
    if not why_relevant:
        issues.append("why_relevant_empty")
    if not benefit:
        issues.append("benefit_empty")
    if not evidence:
        issues.append("evidence_empty")

    if issues:
        domain_cn = _DOMAIN_CN.get(card.get("domain", "ai"), "AI")
        display_name = original_title or title or "未命名信息差"
        card["title"] = f"{display_name}——{domain_cn}领域新信号"
        if not info_gap:
            card["information_gap"] = f"这条来自{domain_cn}领域的信息「{display_name}」值得关注，建议带入对话让 Agent 分析其与你当前系统的关联。"
        if not why_relevant:
            card["why_relevant"] = f"「{display_name}」涉及{domain_cn}方向，与你的信息差 Agent OS 技术路线可能存在交集。"
        if not benefit:
            card["benefit"] = f"理解「{display_name}」可以帮助你判断{domain_cn}方向的技术选型和产品决策。"
        card["_quality_issues"] = issues

    return card


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
    meaningful = [t for t in tags if t.lower() not in
                  ("paper", "arxiv", "cs.lg", "cs.cl", "cs.ai", "cs.cv", "github", "rss", "blog")]
    cn_parts = []
    seen = set()
    for tag in meaningful[:4]:
        cn = _TERM_CN.get(tag.lower(), "")
        if cn and cn not in seen:
            cn_parts.append(cn)
            seen.add(cn)

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
                if w.isupper() and len(w) <= 6:
                    cn_parts.append(w)
                    seen.add(w)

    if not cn_parts:
        domain_cn = _DOMAIN_CN.get(domain, "AI")
        return f"{domain_cn}技术"

    topic = " ".join(cn_parts[:4])
    if len(cn_parts) <= 2:
        domain_cn = _DOMAIN_CN.get(domain, "AI")
        return f"{domain_cn}{topic}"
    return topic


def shorten_title_words(title: str, max_words: int = 8, max_chars: int = 80) -> str:
    """Shorten a title by word boundary, not hard character truncation."""
    if not title:
        return ""
    clean = " ".join(title.split())
    if _contains_chinese(clean):
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars - 1] + "…"
    words = clean.split()
    if len(words) <= max_words and len(clean) <= max_chars:
        return clean
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "…"
    return clean[:max_chars - 1] + "…"


def _extract_entity_name(original_title: str) -> str:
    """Extract a short entity name from the original title for injection into card text."""
    if not original_title:
        return ""
    clean = " ".join(original_title.split())
    if _contains_chinese(clean):
        return shorten_title_words(clean, max_words=10, max_chars=40)
    if ":" in clean:
        parts = clean.split(":", 1)
        main = parts[1].strip().rstrip(".") if len(parts) > 1 else parts[0].strip()
        return shorten_title_words(main, max_words=8, max_chars=60)
    if "/" in clean:
        return shorten_title_words(clean, max_words=6, max_chars=60)
    return shorten_title_words(clean, max_words=8, max_chars=60)


def _generate_chinese_title(original_title: str, source_type: str, tags: list[str], domain: str, relation_type: str = "", source_kind: str = "") -> str:
    clean = " ".join((original_title or "").split())
    if not clean:
        return "未命名信息差"

    if _contains_chinese(clean) and len(clean) <= 64:
        return clean
    if _contains_chinese(clean):
        return clean[:61] + "…"

    # far_domain: never use GitHub-style or Agent-style titles
    if relation_type == "far_domain":
        return _far_domain_title(clean, domain, tags, source_kind)

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
    # Keep entity name visible — wrap original paper name in quotes
    entity = _extract_entity_name(title)
    templates = [
        f"「{entity}」——{domain_cn}领域新研究",
        f"关于{topic}的新论文：「{entity}」",
        f"「{entity}」对{topic}的新探索",
        f"论文「{entity}」带来的{topic}信号",
        f"「{entity}」：{topic}方向的技术新思路",
        f"从「{entity}」看{topic}的演进趋势",
    ]
    idx = hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 64:
        result = result[:61] + "…"
    return result


def _github_title(title: str, domain: str, tags: list[str]) -> str:
    topic = _build_cn_topic(title, tags, domain)
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    entity = _extract_entity_name(title)
    templates = [
        f"值得关注的开源项目：「{entity}」",
        f"「{entity}」：{topic}方向的实用工具",
        f"「{entity}」——{domain_cn}领域的开源方案",
        f"GitHub 新项目「{entity}」与{topic}",
        f"适合{domain_cn}场景的「{entity}」",
    ]
    idx = hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 64:
        result = result[:61] + "…"
    return result


def _far_domain_title(title: str, domain: str, tags: list[str], source_kind: str = "") -> str:
    """Generate a far_domain title — NEVER mention Agent/GitHub/RAG/MCP directly."""
    entity = _extract_entity_name(title)
    topic = _build_cn_topic(title, tags, domain)
    # For bucket_seed far_domain, use clean generic templates
    templates = [
        f"「{entity}」——远域信号捕捉模式",
        f"从「{entity}」看非技术领域的反馈回路",
        f"「{entity}」：弱信号如何被产品化捕捉",
        f"远域启发：「{entity}」中的机会发现思路",
        f"「{entity}」——信号筛选与行动判断",
        f"来自{topic}领域的远域信号：「{entity}」",
    ]
    idx = hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 64:
        result = result[:61] + "…"
    return result


def _generic_cn_title(title: str, domain: str, tags: list[str], source_type: str) -> str:
    topic = _build_cn_topic(title, tags, domain)
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    entity = _extract_entity_name(title)
    templates = [
        f"「{entity}」——值得关注的{domain_cn}信号",
        f"关于{topic}的新信息差：「{entity}」",
        f"从「{entity}」看{domain_cn}的新变化",
        f"{domain_cn}方向的新发现：「{entity}」",
        f"「{entity}」：{topic}领域的最新动态",
        f"一条关于{topic}的高价值信息：「{entity}」",
    ]
    idx = hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 64:
        result = result[:61] + "…"
    return result


def _generate_one_sentence_value(title: str, domain: str, tags: list[str], source_type: str, relation_type: str = "") -> str:
    template_key = "far_domain" if relation_type == "far_domain" else domain
    prefixes = _VALUE_PREFIXES.get(template_key, _VALUE_PREFIXES["ai"])
    idx = hash(title + "value") % len(prefixes)
    keywords = _extract_keywords(title, tags)
    entity = _extract_entity_name(title)
    suffix_templates = [
        f"「{entity}」可能改变你对{keywords}的理解和技术选型。",
        f"「{entity}」提供了一种可复用的思路，能直接改进你的{_DOMAIN_CN.get(domain, 'AI')}模块。",
        f"「{entity}」标记了一个容易被忽略但实际很重要的技术信号。",
        f"「{entity}」把{keywords}和你当前的产品方向连接了起来。",
        f"这条信息的核心变化在于：它可能影响你现有{_DOMAIN_CN.get(domain, 'AI')}系统的设计决策。",
    ]
    suffix_idx = hash(title + "suffix") % len(suffix_templates)
    return prefixes[idx] + suffix_templates[suffix_idx]


def _generate_why_relevant(title: str, domain: str, tags: list[str], interests: list[str], source_type: str, relation_type: str = "") -> str:
    template_key = "far_domain" if relation_type == "far_domain" else domain
    prefixes = _WHY_RELEVANT_PREFIXES.get(template_key, _WHY_RELEVANT_PREFIXES["ai"])
    idx = hash(title + "why") % len(prefixes)
    matched = [t for t in tags if t.lower() in " ".join(interests).lower()]
    entity = _extract_entity_name(title)
    if relation_type == "far_domain":
        suffix = f"它和你当前的技术栈没有直接关系，但价值在于信号捕捉和机会发现的方法。"
    elif matched:
        suffix = f"尤其涉及你关注的 {matched[0]}。"
    else:
        suffix = f"「{entity}」和你的信息差 Agent OS 技术路线有交集。"
    return prefixes[idx].format(title=entity) + suffix


def _generate_benefit(domain: str, tags: list[str], title: str, relation_type: str = "") -> str:
    template_key = "far_domain" if relation_type == "far_domain" else domain
    templates = _BENEFIT_TEMPLATES.get(template_key, _BENEFIT_TEMPLATES["ai"])
    idx = hash(title + str(tags)) % len(templates)
    entity = _extract_entity_name(title)
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    return templates[idx].format(title=entity, domain_cn=domain_cn)


def _generate_information_gap(domain: str, tags: list[str], source_type: str, title: str, relation_type: str = "") -> str:
    template_key = "far_domain" if relation_type == "far_domain" else domain
    templates = _GAP_TEMPLATES.get(template_key, _GAP_TEMPLATES["ai"])
    idx = hash(title + str(tags) + "gap") % len(templates)
    entity = _extract_entity_name(title)
    return templates[idx].format(title=entity)


def _generate_next_action(domain: str, source_type: str, title: str) -> str:
    idx = hash(title + domain + source_type) % len(_NEXT_ACTIONS)
    entity = _extract_entity_name(title)
    return _NEXT_ACTIONS[idx].format(title=entity)


def _generate_chinese_summary(original_title: str, summary: str, domain: str) -> str:
    if summary and len(summary) > 20:
        return summary[:200]
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    return f"这条来自{domain_cn}领域的信息值得进一步研究，可能对你的产品和技术决策有参考价值。"


def _extract_keywords(text: str, tags: list[str]) -> str:
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
