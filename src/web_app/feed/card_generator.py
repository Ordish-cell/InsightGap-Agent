import re
from hashlib import sha256
from typing import Any

from src.web_app.feed.personalization import match_user_facts, render_relevance

_DOMAIN_CN = {
    "agent": "Agent",
    "rag": "RAG",
    "devtools": "开发工具",
    "startup": "创业机会",
    "research": "研究前沿",
    "ai": "AI",
}

# Templates describe the signal or suggest verification, never the reader's identity.
_BENEFIT_TEMPLATES = {
    domain: [
        "「{title}」可作为了解{domain_cn}方向的参考资料，具体价值需结合原文判断。",
        "可从「{title}」中核对方法、适用条件和限制，作为{domain_cn}方向的研究素材。",
    ]
    for domain in (*_DOMAIN_CN, "far_domain")
}

_GAP_TEMPLATES = {
    domain: [
        "围绕「{title}」，值得进一步核对原始证据、适用条件及尚未解决的问题。",
        "「{title}」提供了一条待验证的信息线索，可对照相关来源检查结论与局限。",
    ]
    for domain in (*_DOMAIN_CN, "far_domain")
}

_NEXT_ACTIONS = [
    "把「{title}」带入对话，分析原文中的主要观点与证据。",
    "对「{title}」做一次深度研究，核对来源、适用条件和限制。",
    "保存「{title}」作为后续研究的参考素材。",
    "对照其他来源验证「{title}」的结论。",
]

_VALUE_PREFIXES = {
    domain: ["一条值得核对的信息线索：", "可进一步阅读的资料："]
    for domain in (*_DOMAIN_CN, "far_domain")
}


def generate_feed_card(info_item: Any, score: dict[str, Any], user_profile: Any, semantic_memories: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    domain = (info_item.raw_metadata or {}).get("domain", "ai")
    tags = (info_item.raw_metadata or {}).get("tags", info_item.topics or [])
    source_type = info_item.source_type or "web"
    original_title = info_item.title
    relation_type = score.get("relation_type", "far_domain")
    source_kind = (info_item.raw_metadata or {}).get("source_kind", "")

    chinese_title = _generate_chinese_title(original_title, source_type, tags, domain, relation_type, source_kind)
    one_sentence_value = _generate_one_sentence_value(original_title, domain, tags, source_type, relation_type)
    personalization_evidence = match_user_facts(info_item, user_profile, semantic_memories or [])
    why_relevant = _generate_why_relevant(personalization_evidence)
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
        "personalization_evidence": personalization_evidence,
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
    if not next_action:
        issues.append("next_action_empty")
    if not evidence:
        issues.append("evidence_empty")

    if issues:
        domain_cn = _DOMAIN_CN.get(card.get("domain", "ai"), "AI")
        display_name = original_title or title or "未命名信息差"
        card["title"] = f"{display_name}——{domain_cn}领域新信号"
        if not info_gap:
            card["information_gap"] = f"这条来自{domain_cn}领域的信息「{display_name}」值得关注，可进一步核对原文证据和适用条件。"
        if not why_relevant:
            card["why_relevant"] = render_relevance(card.get("personalization_evidence") or [])
        if not benefit:
            card["benefit"] = f"「{display_name}」可作为了解{domain_cn}方向的参考资料。"
        if not next_action:
            card["next_action"] = f"核对「{display_name}」的原始来源与证据。"
        card["_quality_issues"] = issues

    card["why_you"] = card["why_relevant"]
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
        # Named papers commonly use "ShortName: descriptive subtitle".
        prefix = parts[0].strip()
        main = prefix if len(prefix) <= 24 else parts[1].strip().rstrip(".")
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
    idx = _stable_hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 64:
        result = result[:61] + "…"
    if is_mostly_english(result):
        result = f"关于{topic}的新研究动态"
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
    idx = _stable_hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 64:
        result = result[:61] + "…"
    return result


def _far_domain_title(title: str, domain: str, tags: list[str], source_kind: str = "") -> str:
    """Describe a discovery signal while preserving the source's entity names."""
    entity = _extract_entity_name(title)
    topic = _build_cn_topic(title, tags, domain)
    # For bucket_seed far_domain, use clean generic templates
    templates = [
        f"「{entity}」——待探索的远域信号",
        f"远域阅读线索：「{entity}」",
        f"「{entity}」：可进一步核对的资料",
        f"远域探索：「{entity}」的观点与证据",
        f"「{entity}」——信息来源与适用条件",
        f"来自{topic}领域的远域信号：「{entity}」",
    ]
    idx = _stable_hash(title) % len(templates)
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
    idx = _stable_hash(title) % len(templates)
    result = templates[idx]
    if len(result) > 64:
        result = result[:61] + "…"
    if is_mostly_english(result):
        result = f"关于{topic}的新研究动态"
    return result


def _stable_hash(text: str) -> int:
    """Keep template choices stable across process restarts."""
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:4], "big")


def _generate_one_sentence_value(title: str, domain: str, tags: list[str], source_type: str, relation_type: str = "") -> str:
    template_key = "far_domain" if relation_type == "far_domain" else domain
    prefixes = _VALUE_PREFIXES.get(template_key, _VALUE_PREFIXES["ai"])
    idx = _stable_hash(title + "value") % len(prefixes)
    keywords = _extract_keywords(title, tags)
    entity = _extract_entity_name(title)
    return prefixes[idx] + f"「{entity}」涉及{keywords}，具体观点与证据见原始来源。"


def _generate_why_relevant(evidence: list[dict[str, Any]]) -> str:
    return render_relevance(evidence)


def _generate_benefit(domain: str, tags: list[str], title: str, relation_type: str = "") -> str:
    template_key = "far_domain" if relation_type == "far_domain" else domain
    templates = _BENEFIT_TEMPLATES.get(template_key, _BENEFIT_TEMPLATES["ai"])
    idx = _stable_hash(title + str(tags)) % len(templates)
    entity = _extract_entity_name(title)
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    return templates[idx].format(title=entity, domain_cn=domain_cn)


def _generate_information_gap(domain: str, tags: list[str], source_type: str, title: str, relation_type: str = "") -> str:
    template_key = "far_domain" if relation_type == "far_domain" else domain
    templates = _GAP_TEMPLATES.get(template_key, _GAP_TEMPLATES["ai"])
    idx = _stable_hash(title + str(tags) + "gap") % len(templates)
    entity = _extract_entity_name(title)
    return templates[idx].format(title=entity)


def _generate_next_action(domain: str, source_type: str, title: str) -> str:
    idx = _stable_hash(title + domain + source_type) % len(_NEXT_ACTIONS)
    entity = _extract_entity_name(title)
    return _NEXT_ACTIONS[idx].format(title=entity)


def _generate_chinese_summary(original_title: str, summary: str, domain: str) -> str:
    if summary and len(summary) > 20:
        return summary[:200]
    domain_cn = _DOMAIN_CN.get(domain, "AI")
    return f"这条来自{domain_cn}领域的信息值得进一步研究，具体观点与适用条件需查阅原始来源。"


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
