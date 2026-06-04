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

import re
from collections import defaultdict
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
}

# Category groups that can be reflected into a single summary
_REFLECTABLE_CATEGORIES = {
    "project_goal": "project_goal_summary",
    "tech_stack": "tech_stack_summary",
    "preference": "ui_preference_summary",
    "boundary": "boundary_summary",
    "feed_interest": "feed_interest_summary",
    "workflow_pattern": "workflow_pattern_summary",
}

# Conflict detection keywords — if a new memory contains a "now_use" or
# reversal pattern that conflicts with an existing "do_not_use" memory,
# we supersede the old one.
_CONFLICT_PAIRS = [
    (r"(不要|不做|暂时不做?|不引入|不接|不使?用).*?({term})", "do_not_use"),
    (r"(现在|可以|开始|准备|要).*?(做|引入|接入?|使?用).*?({term})", "now_use"),
    (r"(偏好|喜欢|倾向|选择).*?({term})", "prefer"),
    (r"(不再|不要|放弃|切换).*?(偏好|喜欢|使用|选择).*?({term})", "no_longer_prefer"),
]

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
        existing_semantic = self._get_active_semantic(user_id, db)
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
        # Check for conflicts with existing memories
        newly_saved = result.get("saved", {})
        for mem in newly_saved.get("semantic", []):
            self._detect_and_supersede(user_id, mem, existing_semantic, db)
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

    def supersede_conflicting_memories(
        self, user_id: int, new_memory: dict[str, Any], db: Session | None = None
    ) -> list[dict[str, Any]]:
        """Find and supersede memories that conflict with a new one."""
        if not db:
            return []
        existing = self._get_active_semantic(user_id, db)
        superseded = []
        new_content = str(new_memory.get("content", ""))
        new_category = str(new_memory.get("metadata", {}).get("category", "") or new_memory.get("category", ""))

        for old in existing:
            old_content = str(old.get("content", "") if isinstance(old, dict) else old.content)
            old_meta = old.get("metadata", {}) if isinstance(old, dict) else (old.metadata_json or {})
            old_category = str(old_meta.get("category", ""))

            if old_category != new_category:
                continue
            if self._is_conflict(old_content, new_content):
                if isinstance(old, dict):
                    old_id = old.get("id")
                else:
                    old_id = old.id
                repo = MemoryRepository(db)
                item = repo.get_by_id(old_id)
                if item:
                    old_meta = dict(item.metadata_json or {})
                    old_meta["status"] = "superseded"
                    old_meta["superseded_by"] = new_memory.get("id")
                    old_meta["superseded_at"] = datetime.now(UTC).isoformat()
                    repo.update(item, metadata_json=old_meta)
                    superseded.append(self._to_dict(item))

                # Update new memory with supersedes link
                new_meta = dict(new_memory.get("metadata", {}))
                new_meta["supersedes"] = old_id
                if new_memory.get("id"):
                    new_item = repo.get_by_id(new_memory["id"])
                    if new_item:
                        repo.update(new_item, metadata_json=new_meta)

        return superseded

    def _detect_and_supersede(
        self, user_id: int, new_memory: dict[str, Any],
        existing: list[dict[str, Any]], db: Session | None = None
    ) -> None:
        """Auto-detect and resolve conflicts after saving a new memory."""
        if not db or not new_memory.get("id"):
            return
        self.supersede_conflicting_memories(user_id, new_memory, db)

    def _is_conflict(self, old_content: str, new_content: str) -> bool:
        """Check if two memory contents describe conflicting settings."""
        for pattern, conflict_type in _CONFLICT_PAIRS:
            for term in _ENTITY_TERMS:
                formatted = pattern.format(term=term)
                old_match = re.search(formatted, old_content, re.IGNORECASE)
                new_match = re.search(formatted, new_content, re.IGNORECASE)
                if old_match and new_match:
                    old_type = conflict_type
                    # Check if the new one represents a reversal
                    new_formatted_alt = _CONFLICT_PAIRS[1][0].format(term=term) if len(_CONFLICT_PAIRS) > 1 else ""
                    if conflict_type == "do_not_use" and re.search(new_formatted_alt, new_content, re.IGNORECASE):
                        return True
                    if conflict_type == "prefer" and re.search(
                        _CONFLICT_PAIRS[3][0].format(term=term), new_content, re.IGNORECASE
                    ):
                        return True
        return False

    # ── Reflection ────────────────────────────────────────────────────

    def reflect_user_profile(
        self, user_id: int, db: Session | None = None
    ) -> dict[str, Any]:
        """Merge fragmented semantic memories of the same category into
        summary profile memories. Deterministic — no LLM required."""
        if not db:
            return {"summaries": [], "archived": []}

        active = self._get_active_semantic(user_id, db)
        if len(active) < 8:
            return {"summaries": [], "archived": [], "reason": "not_enough_memories"}

        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for mem in active:
            meta = mem.get("metadata", {}) if isinstance(mem, dict) else (mem.metadata_json or {})
            cat = meta.get("category", "uncategorized")
            by_category[cat].append(mem)

        summaries = []
        archived_ids = []

        for cat, group in by_category.items():
            if len(group) < 4:
                continue
            summary_cat = _REFLECTABLE_CATEGORIES.get(cat)
            if not summary_cat:
                continue

            contents = [m.get("content", "") if isinstance(m, dict) else m.content for m in group]
            summary_content = self._build_category_summary(cat, contents)

            best_importance = max(
                m.get("importance", 0.7) if isinstance(m, dict) else m.importance
                for m in group
            )
            source_ids = [m.get("id") if isinstance(m, dict) else m.id for m in group]

            summary = memory_service.add_with_dedup(
                user_id, summary_content,
                memory_type="semantic",
                importance=min(0.96, best_importance + 0.05),
                metadata={
                    "category": summary_cat, "source": "reflection",
                    "stability": "long_term", "status": "active",
                    "confidence": 0.85,
                    "source_memory_ids": source_ids,
                    "reflected_at": datetime.now(UTC).isoformat(),
                }, db=db,
            )
            if summary:
                summaries.append(summary)

            # Archive original fragments
            for source_id in source_ids:
                repo = MemoryRepository(db)
                item = repo.get_by_id(source_id)
                if item:
                    meta = dict(item.metadata_json or {})
                    meta["status"] = "archived"
                    meta["archived_into"] = summary.get("id") if summary else None
                    repo.update(item, metadata_json=meta)
                    archived_ids.append(source_id)

        return {
            "summaries": summaries,
            "archived": archived_ids,
            "summary_count": len(summaries),
            "archived_count": len(archived_ids),
        }

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
