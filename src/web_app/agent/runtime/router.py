from src.web_app.agent.runtime.state import AgentRoute

RESEARCH_TERMS = ("research", "deep research", "调研", "研究", "深度", "信息差", "机会")
RAG_TERMS = ("rag", "知识库", "资料", "文档", "问答", "引用", "检索")
ARTIFACT_TERMS = ("artifact", "报告", "文稿", "草稿", "生成一份")
SKILL_TERMS = ("skill", "技能", "流程", "沉淀", "复用")


def route_user_input(user_input: str, payload: dict | None = None) -> AgentRoute:
    payload = payload or {}
    forced = payload.get("route") or payload.get("intent")
    if forced in {"research", "rag", "artifact", "skill", "memory", "tool"}:
        return forced
    if payload.get("tool_name"):
        return "tool"
    if payload.get("feed_card_id"):
        return "research"

    text = user_input.lower()
    if any(term in text for term in RESEARCH_TERMS):
        return "research"
    if any(term in text for term in RAG_TERMS):
        return "rag"
    if any(term in text for term in SKILL_TERMS):
        return "skill"
    if any(term in text for term in ARTIFACT_TERMS):
        return "artifact"
    return "memory"
