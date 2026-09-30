"""User Growth Engine — processes behavioral signals into evolving user profiles.

This service does NOT create a "preferences settings page." Instead, it
continuously extracts, refines, supersedes, and reflects user long-term
settings from conversation, feed feedback, skill events, artifacts, and
research activity.

Design principles:
- Deterministic (no LLM required for core operations)
- Idempotent (same signal processed twice should not duplicate memories)
- Supersede-aware (new settings can explicitly override old ones)
- Decay-aware (effective importance degrades for stale memories)
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.services.memory_service import memory_service


# How fast each stability tier decays per 30 days of not being seen
_DECAY_RATES = {
    "temporary": 0.40,
    "session": 0.25,
    "medium_term": 0.10,
    "long_term": 0.03,
}

# Multiplier for active / superseded / archived status
_STATUS_FACTORS = {
    "active": 1.0,
    "superseded": 0.0,
    "archived": 0.25,
    "low_confidence": 0.50,
    "pending": 0.0,
}

_ENTITY_TERMS = [
    "exa", "neo4j", "neo4j", "qdrant", "redis", "mysql", "langgraph", "langchain",
    "fastapi", "vite", "react", "typescript", "python", "pycharm", "openai",
    "anthropic", "deepseek", "codex", "mcp", "rag", "skill", "feed",
]


class UserGrowthService:
    """Unified entry point for all user growth signals."""

    # ── Public signal processors ──────────────────────────────────────

    def process_conversation(
        self,
        user_id: int,
        user_input: str,
        agent_output: str = "",
        page_context: dict[str, Any] | None = None,
        feed_card_context: dict[str, Any] | None = None,
        matched_skill: dict[str, Any] | None = None,
        created_skill_draft: dict[str, Any] | None = None,
        route: str = "",
        db: Session | None = None,
    ) -> dict[str, Any]:
        """Process a conversation turn through the growth engine."""
        result = memory_service.extract_and_save(
            user_id=user_id,
            user_input=user_input,
            agent_output=agent_output,
            page_context=page_context,
            feed_card_context=feed_card_context,
            matched_skill=matched_skill,
            created_skill_draft=created_skill_draft,
            db=db,
        )
        # MemoryService owns scoped comparison and atomic version replacement.
        return result

    def process_feed_feedback(
        self,
        user_id: int,
        card_id: int,
        action: str,
        card_title: str = "",
        card_domain: str = "",
        card_topics: list[str] | None = None,
        db: Session | None = None,
    ) -> dict[str, Any]:
        """Process feed card feedback (save / ignore / useful / not_relevant / research)."""
        saved = []
        topics = card_topics or []

        if action in ("save", "useful"):
            content = f"用户对 FeedCard「{card_title}」标记了 {action}，主题涉及 {', '.join(topics[:3]) or card_domain}。"
            mem = memory_service.add_with_dedup(
                user_id, content, memory_type="episodic",
                importance=0.60, metadata={
                    "category": "feed_feedback", "source": "feed_action",
                    "action": action, "card_id": card_id,
                    "stability": "session", "status": "active",
                }, db=db,
            )
            if mem:
                saved.append(mem)
            # Boost feed_interest for the card's topics
            if topics:
                self._boost_feed_interests(user_id, topics, db)

        elif action in ("ignore", "not_relevant"):
            content = f"用户对 FeedCard「{card_title}」标记了 {action}，对该主题不感兴趣。"
            mem = memory_service.add_with_dedup(
                user_id, content, memory_type="episodic",
                importance=0.55, metadata={
                    "category": "negative_preference", "source": "feed_action",
                    "action": action, "card_id": card_id,
                    "stability": "session", "status": "active",
                }, db=db,
            )
            if mem:
                saved.append(mem)
            if topics:
                self._add_negative_preference(user_id, topics, db)

        elif action == "deep_research":
            content = f"用户从 FeedCard「{card_title}」启动了深度研究。"
            mem = memory_service.add_with_dedup(
                user_id, content, memory_type="episodic",
                importance=0.65, metadata={
                    "category": "research_action", "source": "feed_action",
                    "action": action, "card_id": card_id,
                    "stability": "session", "status": "active",
                }, db=db,
            )
            if mem:
                saved.append(mem)

        return {"action": action, "saved": saved}

    def process_skill_event(
        self,
        user_id: int,
        skill_id: int,
        event: str,  # approve / disable / use_success / use_failure / create
        skill_name: str = "",
        db: Session | None = None,
    ) -> dict[str, Any]:
        """Process skill lifecycle events."""
        saved = []
        if event == "approve":
            content = f"用户批准了 Skill「{skill_name}」(id={skill_id})，确认该工作流可复用。"
            mem = memory_service.add_with_dedup(
                user_id, content, memory_type="episodic",
                importance=0.70, metadata={
                    "category": "skill_approval", "source": "skill_event",
                    "skill_id": skill_id, "stability": "medium_term", "status": "active",
                }, db=db,
            )
            if mem:
                saved.append(mem)
            self._boost_workflow_pattern(user_id, skill_name, db)

        elif event == "disable":
            content = f"用户禁用了 Skill「{skill_name}」(id={skill_id})。"
            mem = memory_service.add_with_dedup(
                user_id, content, memory_type="episodic",
                importance=0.55, metadata={
                    "category": "skill_disable", "source": "skill_event",
                    "skill_id": skill_id, "stability": "session", "status": "active",
                }, db=db,
            )
            if mem:
                saved.append(mem)

        elif event == "use_success":
            content = f"Skill「{skill_name}」(id={skill_id}) 执行成功。"
            mem = memory_service.add_with_dedup(
                user_id, content, memory_type="episodic",
                importance=0.50, metadata={
                    "category": "skill_usage", "source": "skill_event",
                    "skill_id": skill_id, "stability": "session", "status": "active",
                }, db=db,
            )
            if mem:
                saved.append(mem)

        elif event == "use_failure":
            content = f"Skill「{skill_name}」(id={skill_id}) 执行失败。"
            mem = memory_service.add_with_dedup(
                user_id, content, memory_type="episodic",
                importance=0.45, metadata={
                    "category": "skill_failure", "source": "skill_event",
                    "skill_id": skill_id, "stability": "session", "status": "active",
                }, db=db,
            )
            if mem:
                saved.append(mem)

        return {"event": event, "saved": saved}

    def process_research_event(
        self,
        user_id: int,
        research_run_id: str,
        query: str = "",
        status: str = "completed",
        db: Session | None = None,
    ) -> dict[str, Any]:
        """Process research completion events."""
        content = f"用户完成了深度研究：{query[:100]}（状态：{status}）"
        mem = memory_service.add_with_dedup(
            user_id, content, memory_type="episodic",
            importance=0.65 if status == "completed" else 0.40,
            metadata={
                "category": "research_completion", "source": "research_event",
                "research_run_id": research_run_id,
                "stability": "session", "status": "active",
            }, db=db,
        )
        return {"research_run_id": research_run_id, "saved": [mem] if mem else []}

    def process_artifact_event(
        self,
        user_id: int,
        artifact_id: int,
        event: str,  # created / saved / regenerated / deleted
        artifact_title: str = "",
        db: Session | None = None,
    ) -> dict[str, Any]:
        """Process artifact lifecycle events."""
        content = f"用户对 Artifact「{artifact_title}」(id={artifact_id}) 执行了 {event}。"
        importance = 0.55 if event in ("saved", "regenerated") else 0.40
        mem = memory_service.add_with_dedup(
            user_id, content, memory_type="episodic",
            importance=importance, metadata={
                "category": "artifact_event", "source": "artifact_event",
                "artifact_id": artifact_id, "event": event,
                "stability": "session", "status": "active",
            }, db=db,
        )
        return {"event": event, "saved": [mem] if mem else []}

    # ── Supersede ─────────────────────────────────────────────────────

    def supersede_conflicting_memories(self, user_id, new_memory, db=None):
        """Compatibility report; MemoryService performs the atomic replacement."""
        if not db or not new_memory.get("id"):
            return []
        return [self._to_dict(item) for item in MemoryRepository(db).list_by_user(user_id)
                if (item.metadata_json or {}).get("superseded_by") == new_memory["id"]]

    # ── Reflection ────────────────────────────────────────────────────

    def reflect_user_profile(self, user_id, db=None, *, conversation_id=None):
        return memory_service.consolidate_memory(user_id, db, conversation_id=conversation_id)

    def _build_category_summary(self, category: str, contents: list[str]) -> str:
        """Deterministic summary from same-category memory contents."""
        # Extract key noun phrases from all contents
        all_text = " ".join(contents)
        # Take the most complete/longest content as base, or concatenate key points
        if category == "project_goal":
            longest = max(contents, key=len)
            return f"用户画像总结：{longest}"
        if category == "tech_stack":
            techs = set()
            for c in contents:
                for term in _ENTITY_TERMS:
                    if term.lower() in c.lower():
                        techs.add(term if term[0].isupper() else term.title() if term in ("fastapi", "vite", "react", "typescript", "python") else term.upper() if term in ("rag", "mcp") else term)
            tech_list = "、".join(sorted(techs)[:10]) if techs else "多种技术"
            return f"用户技术栈总结：{tech_list}。"
        if category in ("preference", "ui_preference"):
            return f"用户偏好总结：{'；'.join(contents[:4])}"
        if category == "boundary":
            return f"用户当前边界总结：{'；'.join(contents[:4])}"
        if category == "feed_interest":
            return f"用户信息兴趣总结：{'；'.join(contents[:4])}"
        if category == "workflow_pattern":
            return f"用户任务模式总结：{'；'.join(contents[:4])}"
        return f"用户画像总结（{category}）：{'；'.join(contents[:3])}"

    # ── Effective importance with decay ───────────────────────────────

    def compute_effective_importance(self, memory: dict[str, Any]) -> float:
        """Compute effective importance factoring in recency decay,
        evidence boost, and status."""
        base = float(memory.get("importance", 0.5))
        meta = memory.get("metadata", {}) if isinstance(memory, dict) else getattr(memory, "metadata_json", {})
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        stability = meta.get("stability", "medium_term")
        status = meta.get("status", "active")
        evidence_count = int(meta.get("evidence_count", 1))
        last_seen = meta.get("last_seen_at", "")

        # Decay by recency
        decay = _DECAY_RATES.get(stability, 0.10)
        days_since = 0
        if last_seen:
            try:
                if isinstance(last_seen, str):
                    last_seen_dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                    days_since = (datetime.now(UTC) - last_seen_dt).days
            except (ValueError, TypeError):
                pass
        periods = max(0, days_since // 30)
        recency_factor = max(0.15, 1.0 - decay * periods)

        # Evidence boost
        evidence_factor = min(1.20, 1.0 + 0.04 * (evidence_count - 1))

        # Status factor
        status_factor = _STATUS_FACTORS.get(status, 0.50)

        effective = base * recency_factor * evidence_factor * status_factor
        return round(min(1.0, max(0.0, effective)), 4)

    def get_memories_with_effective_importance(
        self, user_id: int, db: Session | None = None,
        memory_type: str | None = None, min_effective: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Return memories with computed effective importance, sorted by it."""
        if not db:
            return []
        raw = memory_service.search_memory(user_id, memory_type=memory_type, db=db)
        enriched = []
        for mem in raw:
            mem_dict = dict(mem) if not isinstance(mem, dict) else mem
            mem_dict["effective_importance"] = self.compute_effective_importance(mem_dict)
            if mem_dict["effective_importance"] >= min_effective:
                enriched.append(mem_dict)
        enriched.sort(key=lambda m: m["effective_importance"], reverse=True)
        return enriched

    # ── Dynamic preference profile for GSSC ───────────────────────────

    def build_dynamic_preference_profile(
        self, user_id: int, db: Session | None = None,
        route: str = "chat",
    ) -> dict[str, Any]:
        """Build a dynamic user profile for GSSC consumption.
        This augments the static UserProfile with live memory data."""
        active_semantic = self.get_memories_with_effective_importance(
            user_id, db, memory_type="semantic", min_effective=0.25
        )
        recent_episodic = self.get_memories_with_effective_importance(
            user_id, db, memory_type="episodic", min_effective=0.30
        )[:5]

        # Route-specific preference extraction
        preference_texts = []
        for mem in active_semantic[:8]:
            meta = mem.get("metadata", {})
            route_scope = meta.get("route_scope", [])
            if not route_scope or route in route_scope:
                preference_texts.append(mem.get("content", ""))

        # Build conversation summary from recent episodic memories
        episodic_texts = [m.get("content", "") for m in recent_episodic[:5]]

        return {
            "dynamic_goals": [m.get("content") for m in active_semantic if m.get("metadata", {}).get("category") == "project_goal"][:2],
            "dynamic_preferences": [m.get("content") for m in active_semantic if m.get("metadata", {}).get("category") in ("preference", "ui_preference")][:3],
            "dynamic_boundaries": [m.get("content") for m in active_semantic if m.get("metadata", {}).get("category") == "boundary"][:2],
            "dynamic_interests": [m.get("content") for m in active_semantic if m.get("metadata", {}).get("category") in ("feed_interest", "workflow_pattern")][:3],
            "recent_activity": episodic_texts[:3],
            "active_memory_count": len(active_semantic),
            "preference_summary": "；".join(preference_texts[:5]),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_active_semantic(self, user_id: int, db: Session | None) -> list[dict[str, Any]]:
        if not db:
            return []
        raw = memory_service.search_memory(user_id, memory_type="semantic", db=db)
        return [
            m for m in raw
            if (m.get("metadata", {}) if isinstance(m, dict) else (m.metadata_json or {})).get("status", "active") != "superseded"
        ]

    def _boost_feed_interests(self, user_id: int, topics: list[str], db: Session | None = None) -> None:
        topic_cn = ", ".join(topics[:4])
        memory_service.add_with_dedup(
            user_id,
            f"用户通过 Feed 反馈表现出对以下主题的兴趣：{topic_cn}。",
            memory_type="semantic", importance=0.65,
            metadata={
                "category": "feed_interest", "source": "feed_action",
                "stability": "medium_term", "status": "active",
                "evidence_count": 1,
            }, db=db,
        )

    def _add_negative_preference(self, user_id: int, topics: list[str], db: Session | None = None) -> None:
        topic_cn = ", ".join(topics[:3])
        memory_service.add_with_dedup(
            user_id,
            f"用户对以下主题表现出负面偏好：{topic_cn}。",
            memory_type="semantic", importance=0.55,
            metadata={
                "category": "negative_preference", "source": "feed_action",
                "stability": "medium_term", "status": "active",
                "negative": True,
            }, db=db,
        )

    def _boost_workflow_pattern(self, user_id: int, skill_name: str, db: Session | None = None) -> None:
        memory_service.add_with_dedup(
            user_id,
            f"用户确认了一个可复用工作流：{skill_name}。",
            memory_type="semantic", importance=0.75,
            metadata={
                "category": "workflow_pattern", "source": "skill_event",
                "stability": "long_term", "status": "active",
            }, db=db,
        )

    def _to_dict(self, item) -> dict[str, Any]:
        return {
            "id": item.id, "user_id": item.user_id,
            "content": item.content, "memory_type": item.memory_type,
            "importance": item.importance,
            "metadata": item.metadata_json or {},
        }


user_growth_service = UserGrowthService()
