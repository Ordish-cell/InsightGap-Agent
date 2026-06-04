import re
from typing import Any


class MemoryExtractor:
    """Deterministic memory extraction from Agent conversations.

    Extracts semantic (long-term settings), episodic (specific events),
    and working (current context) memories without requiring an LLM.
    """

    def extract(
        self,
        user_input: str,
        agent_output: str = "",
        page_context: dict[str, Any] | None = None,
        feed_card_context: dict[str, Any] | None = None,
        matched_skill: dict[str, Any] | None = None,
        created_skill_draft: dict[str, Any] | None = None,
        user_profile: dict[str, Any] | None = None,
        recent_memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        text = f"{user_input} {agent_output}"
        result: dict[str, list[dict[str, Any]]] = {
            "working_memories": [],
            "episodic_memories": [],
            "semantic_memories": [],
        }

        result["working_memories"] = self._extract_working(page_context, feed_card_context)
        result["episodic_memories"] = self._extract_episodic(user_input, agent_output, feed_card_context, matched_skill, created_skill_draft)
        result["semantic_memories"] = self._extract_semantic(text, user_input, user_profile, recent_memories)
        result["should_consolidate"] = bool(result["semantic_memories"]) or bool(result["episodic_memories"] and any(m.get("importance", 0) >= 0.7 for m in result["episodic_memories"]))

        return result

    def _extract_working(
        self,
        page_context: dict[str, Any] | None,
        feed_card_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        result = []
        page = (page_context or {}).get("page", "")
        if page:
            result.append({
                "content": f"当前页面：{page}",
                "importance": 0.3,
                "category": "working_context",
                "source": "page_context",
                "confidence": 0.95,
            })
        if feed_card_context:
            title = feed_card_context.get("title", "")
            card_id = feed_card_context.get("id", "")
            result.append({
                "content": f"当前选中 FeedCard：{title} (id={card_id})",
                "importance": 0.4,
                "category": "working_context",
                "source": "feed_card",
                "confidence": 0.95,
            })
        return result

    def _extract_episodic(
        self,
        user_input: str,
        agent_output: str,
        feed_card_context: dict[str, Any] | None,
        matched_skill: dict[str, Any] | None,
        created_skill_draft: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        result = []
        if matched_skill:
            result.append({
                "content": f"Agent 匹配到已有 Skill：{matched_skill.get('name', '')}（评分 {matched_skill.get('match_score', 0)}）",
                "importance": 0.55,
                "category": "skill_match",
                "source": "agent_runtime",
                "confidence": 0.85,
            })
        if created_skill_draft:
            result.append({
                "content": f"Agent 创建了新的 Skill 草稿：{created_skill_draft.get('name', '')}",
                "importance": 0.65,
                "category": "skill_creation",
                "source": "agent_runtime",
                "confidence": 0.85,
            })
        if feed_card_context and any(kw in user_input for kw in ["研究", "分析", "深入", "research", "deep"]):
            result.append({
                "content": f"用户对 FeedCard「{feed_card_context.get('title', '')}」启动了深度研究",
                "importance": 0.55,
                "category": "research_action",
                "source": "home_chat",
                "confidence": 0.80,
            })

        feedback_memories = self._extract_feedback_episodic(user_input, agent_output)
        result.extend(feedback_memories)
        return result

    def _extract_feedback_episodic(self, user_input: str, agent_output: str) -> list[dict[str, Any]]:
        result = []
        feedback_patterns = [
            (r"(首页|feed|Feed).*?(英文|中文|标题|展示|显示)", "用户对首页 FeedCard 的展示语言和格式提出了反馈。", 0.80, "ui_feedback"),
            (r"(不喜欢|不要|讨厌|受不了).*?(英文|模板|重复|一样)", "用户明确表达了负面偏好：不喜欢模板化或重复的内容展示。", 0.85, "negative_feedback"),
            (r"(太慢|太卡|性能|速度).*?(问题|慢|卡)", "用户反馈了性能或响应速度问题。", 0.75, "performance_feedback"),
            (r"(能不能|可以|希望|想要|建议).*?(增加|加上|改进|优化|改成)", "用户提出了功能改进建议。", 0.70, "feature_request"),
        ]
        for pattern, content, importance, category in feedback_patterns:
            if re.search(pattern, user_input + agent_output):
                result.append({
                    "content": content,
                    "importance": importance,
                    "category": category,
                    "source": "home_chat",
                    "confidence": 0.75,
                })
        return result

    def _extract_semantic(
        self,
        full_text: str,
        user_input: str,
        user_profile: dict[str, Any] | None,
        recent_memories: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        result = []

        result.extend(self._extract_project_goals(full_text))
        result.extend(self._extract_tech_stack(full_text))
        result.extend(self._extract_boundaries(full_text))
        result.extend(self._extract_preferences(full_text))
        result.extend(self._extract_feed_interests(full_text))

        casual = self._is_casual_chat(user_input)
        if casual:
            result = [m for m in result if m.get("importance", 0) >= 0.80]
            for m in result:
                m["importance"] = min(m["importance"], 0.50)

        return result

    def _extract_project_goals(self, text: str) -> list[dict[str, Any]]:
        result = []
        patterns = [
            (r"我正在?(开发|做|构建|打造|搭建).*?(信息差|Agent OS|agent os|信息差.*?agent|agent.*?系统)", "用户正在开发基于 Open Deep Research 二开的信息差 Agent OS。", 0.90, "project_goal"),
            (r"(长期|最终|核心).*?(目标|方向|闭环).*?(信息差|Agent|Feed|Skill|Memory)", "用户长期目标是构建信息差 Agent OS 闭环：Feed → Agent 研究 → Artifact → Memory/Skill。", 0.88, "project_goal"),
            (r"(二开|基于|fork).*?open.?deep.?research", "用户基于 GitHub Open Deep Research 项目进行二次开发。", 0.85, "project_goal"),
        ]
        for pattern, content, importance, category in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result.append({
                    "content": content,
                    "importance": importance,
                    "category": category,
                    "source": "home_chat",
                    "confidence": 0.80,
                })
        return result

    def _extract_tech_stack(self, text: str) -> list[dict[str, Any]]:
        result = []
        tech_indicators = {
            "fastapi": ("FastAPI", 0.82),
            "vite": ("Vite", 0.82),
            "react": ("React", 0.82),
            "langgraph": ("LangGraph", 0.85),
            "langchain": ("LangChain", 0.85),
            "mysql": ("MySQL", 0.82),
            "redis": ("Redis", 0.82),
            "qdrant": ("Qdrant", 0.85),
            "typescript": ("TypeScript", 0.82),
            "pycharm": ("PyCharm", 0.80),
            "python": ("Python", 0.80),
        }
        found = []
        for key, (name, importance) in tech_indicators.items():
            if key in text.lower():
                found.append(name)
        if len(found) >= 3:
            result.append({
                "content": f"用户技术栈包括：{', '.join(found)}。",
                "importance": 0.85,
                "category": "tech_stack",
                "source": "home_chat",
                "confidence": 0.80,
            })
        elif found:
            result.append({
                "content": f"用户使用了 {', '.join(found)}。",
                "importance": 0.75,
                "category": "tech_stack",
                "source": "home_chat",
                "confidence": 0.75,
            })
        return result

    def _extract_boundaries(self, text: str) -> list[dict[str, Any]]:
        result = []
        boundary_patterns = [
            (r"(不要|不做|暂时不做?|不引入|不接|不使?用).*?(exa|neo4j|neo4j|真实电脑|真实邮件)", "用户当前阶段不希望引入 Exa、Neo4j 或真实电脑操作。", 0.88, "boundary"),
            (r"(不要|不允许).*?(暴露.*?json|原始.*?json)", "用户不希望将原始 JSON 暴露给普通用户。", 0.85, "boundary"),
            (r"(不要|不做).*?(重写.*?agent.*?runtime|破坏.*?api)", "用户不希望重写 Agent Runtime 或破坏现有 API。", 0.85, "boundary"),
            (r"不要修改.*?open.?deep.?research", "用户要求不要修改 src/open_deep_research/ 目录。", 0.90, "boundary"),
        ]
        for pattern, content, importance, category in boundary_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result.append({
                    "content": content,
                    "importance": importance,
                    "category": category,
                    "source": "home_chat",
                    "confidence": 0.80,
                })
        return result

    def _extract_preferences(self, text: str) -> list[dict[str, Any]]:
        result = []
        pref_patterns = [
            (r"(中文|中文化|中文.*?产品化|中文.*?文案)", "用户偏好首页 FeedCard 使用中文展示，文案应简练且有产品感。", 0.88, "preference"),
            (r"(简练|简洁|简短|不要长|不要大段)", "用户偏好简洁的表达方式，不喜欢大段文字。", 0.82, "preference"),
            (r"(首页.*?是|首页.*?布局|home.*?page)", "用户认为首页应该是 Feed + Agent Chat 的组合，不是普通聊天机器人。", 0.85, "preference"),
            (r"(Codex|codex).*?(风格|布局|风格)", "用户偏好 Codex 风格的界面布局。", 0.82, "preference"),
            (r"(产品化|产品感|产品.*?卡片|产品.*?表达)", "用户偏好产品化的表达方式而非学术或技术文档风格。", 0.85, "preference"),
        ]
        for pattern, content, importance, category in pref_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result.append({
                    "content": content,
                    "importance": importance,
                    "category": category,
                    "source": "home_chat",
                    "confidence": 0.80,
                })
        return result

    def _extract_feed_interests(self, text: str) -> list[dict[str, Any]]:
        result = []
        interest_keywords = {
            "agent": "Agent 技术",
            "rag": "RAG 检索增强",
            "memory": "Memory 记忆系统",
            "skill": "Skill 复用",
            "mcp": "MCP 协议",
            "deep research": "Deep Research",
            "feed": "Feed 推荐",
            "开发工具": "开发工具",
            "ai 产品": "AI 产品机会",
        }
        found = []
        for key, label in interest_keywords.items():
            if key in text.lower():
                found.append(label)
        if len(found) >= 3:
            result.append({
                "content": f"用户 Feed 兴趣主题包括：{', '.join(found)}。",
                "importance": 0.80,
                "category": "feed_interest",
                "source": "home_chat",
                "confidence": 0.75,
            })
        return result

    def _is_casual_chat(self, user_input: str) -> bool:
        casual_patterns = [
            r"^(你好|hi|hello|hey)[\s!！。.,，]*$",
            r"^(谢谢|thanks|thank you|3q)[\s!！。.,，]*$",
            r"^(好的|ok|okay|行|可以|明白了|知道了)[\s!！。.,，]*$",
            r"^(再见|拜拜|bye|goodbye)[\s!！。.,，]*$",
            r"^(早上好|下午好|晚上好|早安|晚安)[\s!！。.,，]*$",
            r"^(今天天气|今天.*?怎么样)[\s!！。.,，]*$",
        ]
        return any(re.search(p, user_input.strip(), re.IGNORECASE) for p in casual_patterns)


memory_extractor = MemoryExtractor()
