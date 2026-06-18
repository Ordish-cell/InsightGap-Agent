from __future__ import annotations

from src.web_app.agent.runtime.node_groups.base import *
from src.web_app.agent.runtime.schemas import StateDelta
from src.web_app.agent.runtime.state_delta import record_node_result


class ReadNodesMixin:
    async def parallel_prefetch(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        state = await run_parallel_prefetch(state, self.db, self.payload)
        append_status_step(
            state,
            key="parallel_prefetch",
            node_name="parallel_prefetch",
            detail="已并行预取知识库、记忆、技能和图谱上下文",
            extra={
                "elapsed_ms": state.get("prefetch_elapsed_ms", 0),
                "warnings": state.get("prefetch_warnings", []),
                "sources": list((state.get("prefetch_results") or {}).keys()),
            },
        )
        emit_visible_thought(self.db, state, "parallel_prefetch", stream_queue=self._stream_queue)
        record_step(
            self.db,
            state["run_id"],
            "parallel_prefetch",
            "prefetch",
            {"route": state.get("route"), "user_input": state.get("user_input", "")},
            {
                "prefetch_results": state.get("prefetch_results", {}),
                "prefetch_warnings": state.get("prefetch_warnings", []),
                "prefetch_elapsed_ms": state.get("prefetch_elapsed_ms", 0),
            },
        )
        record_node_result(
            state,
            node="parallel_prefetch",
            delta=StateDelta(
                updates={
                    "prefetch_results": state.get("prefetch_results", {}),
                    "prefetch_warnings": state.get("prefetch_warnings", []),
                    "prefetch_elapsed_ms": state.get("prefetch_elapsed_ms", 0),
                    "prefetch_agent_results": state.get("prefetch_agent_results", []),
                },
                warnings=list(state.get("prefetch_warnings", []) or []),
                metadata={"source": "parallel_prefetch"},
            ),
            summary="Completed read-only parallel prefetch.",
            elapsed_ms=int(state.get("prefetch_elapsed_ms") or 0),
        )
        return state

    async def parallel_read_stage(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        state = await run_parallel_read_stage(state, self, self.payload)
        append_status_step(
            state,
            key="parallel_read_stage",
            node_name="parallel_read_stage",
            detail="已并行准备上下文、Skill 匹配和 RAG evidence",
            extra={
                "elapsed_ms": state.get("parallel_read_elapsed_ms", 0),
                "warnings": state.get("parallel_read_warnings", []),
                "branches": state.get("parallel_read_branch_timings", {}),
            },
        )
        emit_visible_thought(self.db, state, "parallel_read_stage", stream_queue=self._stream_queue)
        record_step(
            self.db,
            state["run_id"],
            "parallel_read_stage",
            "parallel_read",
            {"route": state.get("route"), "route_plan": (state.get("route_plan") or {}).get("route", [])},
            {
                "parallel_read_results": state.get("parallel_read_results", {}),
                "parallel_read_warnings": state.get("parallel_read_warnings", []),
                "parallel_read_elapsed_ms": state.get("parallel_read_elapsed_ms", 0),
                "parallel_read_branch_timings": state.get("parallel_read_branch_timings", {}),
            },
        )
        record_node_result(
            state,
            node="parallel_read_stage",
            delta=StateDelta(
                updates={
                    "parallel_read_results": state.get("parallel_read_results", {}),
                    "parallel_read_warnings": state.get("parallel_read_warnings", []),
                    "parallel_read_elapsed_ms": state.get("parallel_read_elapsed_ms", 0),
                    "parallel_read_branch_timings": state.get("parallel_read_branch_timings", {}),
                },
                warnings=list(state.get("parallel_read_warnings", []) or []),
                metadata={"source": "parallel_read_stage"},
            ),
            summary="Completed parallel read preparation.",
            elapsed_ms=int(state.get("parallel_read_elapsed_ms") or 0),
        )
        return state

    async def context_builder(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        route = state.get("route", "chat")
        user_input = state.get("user_input", "")
        answer_mode = (state.get("route_plan") or {}).get("answer_mode", state.get("answer_mode", "chat"))
        is_conversation_recall = answer_mode == "conversation_recall"
        prefetch_results = state.get("prefetch_results") or {}
        profile = ProfileRepository(self.db).get_or_create_default(state["user_id"])
        query_memories: list[dict[str, Any]] = []
        baseline_memories: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        if is_conversation_recall:
            memory_search_backend = "skipped_conversation_recall"
            memory_qdrant_hits = 0
        else:
            memory_prefetch = prefetch_results.get("memory") if isinstance(prefetch_results.get("memory"), dict) else None
            if memory_prefetch is not None and "items" in memory_prefetch:
                query_memories = list(memory_prefetch.get("items") or [])
                memory_search_backend = memory_prefetch.get("backend", "prefetch")
                memory_qdrant_hits = memory_prefetch.get("qdrant_hits", 0)
            else:
                query_memories = memory_service.search_memory(
                    state["user_id"], user_input, min_importance=0.2, db=self.db, limit=5,
                )
                memory_search_backend = getattr(memory_service, "_last_search_backend", "unknown")
                memory_qdrant_hits = getattr(memory_service, "_last_qdrant_hits", 0)
        page_context = self.payload.get("page_context") or state.get("page_context") or {}
        feed_card_id = self.payload.get("feed_card_id") or page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
        feed_card_context = self._load_feed_card_context(state["user_id"], feed_card_id)

        # ── Load conversation history from agent_chat_messages ────────
        conversation_history_text = ""
        conversation_history_debug: dict[str, Any] = {}
        recent_messages: list[Any] = []
        try:
            conversation_id = state.get("conversation_id")
            if conversation_id and state.get("user_id"):
                message_repo = AgentChatMessageRepository(self.db)
                recent_messages = message_repo.list_recent_by_conversation(
                    user_id=state["user_id"],
                    conversation_id=conversation_id,
                    limit=12,
                )
                conversation_history_text = self._format_recent_chat_messages_for_context(recent_messages)
                conversation_history_debug = {
                    "state_user_id": state["user_id"],
                    "state_conversation_id": conversation_id,
                    "recent_chat_messages_count": len(recent_messages),
                    "recent_chat_message_preview": [
                        f"{m.role}: {(m.content or '')[:80]}"
                        for m in recent_messages[:5]
                    ],
                    "has_conversation_history_text": bool(conversation_history_text),
                }
            else:
                conversation_history_debug = {
                    "conversation_history_empty_reason": "missing_user_id_or_conversation_id",
                    "state_user_id": state.get("user_id"),
                    "state_conversation_id": None,
                }
        except Exception as exc:
            logger.exception(
                "context_builder.load_conversation_history_failed",
                extra={
                    "user_id": state.get("user_id"),
                    "conversation_id": state.get("conversation_id"),
                    "error": str(exc),
                },
            )
            conversation_history_debug = {
                "conversation_history_empty_reason": "repository_error",
                "error": str(exc),
            }

        # Generate conversation summary from recent agent steps
        conversation_summary = self._build_conversation_summary(state["run_id"])

        # Generate checkpoint summary from current state
        checkpoint_summary = self._build_checkpoint_summary(state)

        # Get dynamic preferences from user growth engine
        dynamic_prefs = user_growth_service.build_dynamic_preference_profile(
            state["user_id"], self.db, route=route,
        )

        # ── RAG Evidence (lightweight, no LLM) ──────────────────────
        rag_evidence: list[dict[str, Any]] = []
        rag_prefetch = prefetch_results.get("rag") if isinstance(prefetch_results.get("rag"), dict) else None
        if is_conversation_recall:
            rag_evidence = []
        elif rag_prefetch is not None and "evidence" in rag_prefetch:
            rag_evidence = list(rag_prefetch.get("evidence") or [])
        else:
            try:
                rag_evidence = rag_service.search_evidence(
                    state["user_id"], user_input, limit=5, score_threshold=0.3,
                )
            except Exception:
                pass  # RAG failure must not block the pipeline

        # ── Apply MEMORY_CONTEXT_POLICY: filter by answer_mode ──────
        from src.web_app.context.builder import MEMORY_CONTEXT_POLICY as _MCP
        allowed_categories = _MCP.get(answer_mode, _MCP.get("chat", set()))
        if not is_conversation_recall:
            baseline_memories = memory_service.get_baseline_memories(
                state["user_id"],
                db=self.db,
                categories=allowed_categories,
                min_importance=0.75,
                limit=6,
            )
            memories = self._dedupe_memories([*baseline_memories, *query_memories])
        if allowed_categories and memories:
            memories = [
                m for m in memories
                if (m.get("metadata") or {}).get("category", "") in allowed_categories
                or (m.get("metadata") or {}).get("category", "") == ""  # uncategorized passes through
            ]
        if is_conversation_recall:
            baseline_memories = []
            query_memories = []
            memories = []

        # ── Format memories as readable text blocks ─────────────────
        memory_text = self._format_memories_for_context(memories)

        # ── Format profile as readable text ─────────────────────────
        profile_text = self._format_profile_for_context(profile)

        # ── Format RAG evidence as readable text ────────────────────
        evidence_text = self._format_rag_evidence_for_context(rag_evidence)

        graph_context_text = ""
        graph_context_debug: dict[str, Any] = {}
        graph_prefetch = prefetch_results.get("graph") if isinstance(prefetch_results.get("graph"), dict) else None
        if is_conversation_recall:
            graph_context_debug = {"skipped": True, "reason": "conversation_recall_uses_conversation_history_only"}
        elif graph_prefetch is not None and "context" in graph_prefetch:
            graph_context_text = str(graph_prefetch.get("context") or "")
            graph_context_debug = dict(graph_prefetch.get("debug") or {})
        else:
            try:
                from src.web_app.services.graph_context_service import graph_context_service
                graph_context_text = graph_context_service.get_context(
                    user_id=state["user_id"],
                    query=user_input,
                    route=route,
                )
                graph_context_debug = getattr(graph_context_service, "last_debug", {}) or {}
            except Exception:
                logger.warning("context_builder.graph_context_failed", exc_info=True)
                graph_context_debug = {"fallback": True, "warning": "graph_context_failed"}

        conversation_recall_context = self._build_conversation_recall_context(
            recent_messages,
            current_user_input=user_input,
        )
        state["conversation_recall_context"] = conversation_recall_context
        state["memory_context"] = {
            "loader": "memory_context_loader",
            "read_only": True,
            "skipped": is_conversation_recall,
            "skip_reason": "conversation_recall_uses_conversation_history_only" if is_conversation_recall else "",
            "baseline_profile_memory": baseline_memories,
            "query_memory": query_memories,
            "items": memories,
            "backend": memory_search_backend,
            "qdrant_hits": memory_qdrant_hits,
        }

        builder = ContextBuilder(route=route)
        context_text, gssc_debug = builder.build_with_debug({
            "task": user_input,
            "route": route,
            "profile": "" if is_conversation_recall else profile_text,
            "memory": memory_text,
            "graph_context": graph_context_text,
            "evidence": evidence_text,
            "feed_card": {} if is_conversation_recall else feed_card_context,
            "page_context": {} if is_conversation_recall else page_context,
            "conversation_history": conversation_history_text,
            "conversation_summary": "" if is_conversation_recall else conversation_summary,
            "checkpoint_summary": "" if is_conversation_recall else checkpoint_summary,
            "dynamic_preferences": "" if is_conversation_recall else dynamic_prefs.get("preference_summary", ""),
            "output_contract": "Return structured status, final_output, artifacts, memory_updates, skill_drafts, and evidence when available.",
        })
        # Merge with existing context (don't overwrite fields set by earlier nodes)
        existing_context = state.get("context") or {}
        state["context"] = {
            **existing_context,
            "gssc_context": context_text,
            "gssc_debug": gssc_debug,
            "memory_count": len(memories),
            "memory_items": memories,
            "memory_context": state["memory_context"],
            "conversation_recall_context": conversation_recall_context,
            "graph_context": graph_context_text,
            "graph_context_debug": graph_context_debug,
            "feed_card": feed_card_context,
            "page_context": page_context,
            "conversation_history": conversation_history_text,
            "conversation_summary": conversation_summary,
            "checkpoint_summary": checkpoint_summary,
            "profile": {"segment": profile.segment, "goals": profile.goals, "interests": profile.explicit_interests},
            "dynamic_preferences": dynamic_prefs.get("preference_summary", ""),
            "rag_evidence": rag_evidence,
        }
        state["rag_evidence"] = rag_evidence
        record_step(self.db, state["run_id"], "context_builder", "context",
                    {"route": route, "feed_card_id": feed_card_id},
                    {"memory_count": len(memories), "feed_card_loaded": bool(feed_card_context),
                     "rag_evidence_count": len(rag_evidence),
                     "gssc_debug": gssc_debug, "context": context_text,
                     "graph_context_debug": graph_context_debug,
                     "recent_chat_messages_count": conversation_history_debug.get("recent_chat_messages_count", 0),
                     "has_conversation_history_text": conversation_history_debug.get("has_conversation_history_text", False),
                     "has_conversation_history_section": "Conversation History" in context_text,
                     "memory_search_backend": memory_search_backend,
                     "memory_qdrant_hits": memory_qdrant_hits,
                     "memory_context_loader_read_only": True,
                     "conversation_recall_memory_skipped": is_conversation_recall,
                     **conversation_history_debug})
        append_pipeline_step(
            state,
            "load_conversation_context",
            detail=f"loaded {conversation_history_debug.get('recent_chat_messages_count', 0)} recent messages",
            extra={
                "message_count": conversation_history_debug.get("recent_chat_messages_count", 0),
                "has_history": bool(conversation_history_text),
            },
        )
        if not is_conversation_recall:
            append_pipeline_step(
                state,
                "load_memory_context",
                detail=f"loaded {len(memories)} memory items",
                extra={
                    "memory_count": len(memories),
                    "baseline_count": len(baseline_memories),
                    "query_count": len(query_memories),
                    "read_only": True,
                    "backend": memory_search_backend,
                },
            )
        append_status_step(
            state,
            key="context_builder",
            node_name="context_builder",
            detail=f"已选择 {len(gssc_debug.get('selected_sources', []))} 类上下文，记忆 {len(memories)} 条，RAG {len(rag_evidence)} 条",
            extra={
                "selected_sources": gssc_debug.get("selected_sources", []),
                "dropped_sources": gssc_debug.get("dropped_sources", []),
                "token_budget_used": gssc_debug.get("token_budget_used", 0),
                "memory_count": len(memories),
                "rag_evidence_count": len(rag_evidence),
                "feed_card_loaded": bool(feed_card_context),
            },
        )
        emit_visible_thought(self.db, state, "context_builder", stream_queue=self._stream_queue)
        if feed_card_id and not feed_card_context:
            record_step(self.db, state["run_id"], "feed_card_context", "load_context",
                        {"feed_card_id": feed_card_id},
                        {"loaded": False, "reason": "not_found_or_forbidden"}, status="failed")
        return state

    async def skill_matcher(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        if self.payload.get("use_existing_skills", True) is False or self.payload.get("auto_skill", True) is False:
            record_step(self.db, state["run_id"], "skill_matcher", "match_skill", {"enabled": False}, {"matched": False, "reason": "disabled_by_request"})
            return state
        prefetch_skill = ((state.get("prefetch_results") or {}).get("skill") or {})
        if isinstance(prefetch_skill, dict) and ("matched_skill" in prefetch_skill or "candidate_skills" in prefetch_skill):
            result = prefetch_skill.get("raw") or {
                "matched_skill": prefetch_skill.get("matched_skill"),
                "candidate_skills": prefetch_skill.get("candidate_skills", []),
            }
        else:
            result = skill_service.match_skill(state.get("user_input", ""), state["user_id"], self.db, state.get("context", {}))
        state["matched_skill"] = result.get("matched_skill")
        state["candidate_skills"] = result.get("candidate_skills", [])
        if state.get("matched_skill"):
            state["context"]["applied_skill"] = state["matched_skill"]
            state["context"]["gssc_context"] = "\n\n".join([state["context"].get("gssc_context", ""), self._skill_context_block(state["matched_skill"])])
        record_step(
            self.db,
            state["run_id"],
            "skill_matcher",
            "match_skill",
            {"user_input": state.get("user_input", ""), "feed_card_id": (state.get("context") or {}).get("feed_card", {}).get("id")},
            {"matched_skill": state.get("matched_skill"), "candidate_skills": state.get("candidate_skills", [])},
        )
        append_status_step(
            state,
            key="skill_matcher",
            node_name="skill_matcher",
            detail="命中可复用 Skill" if state.get("matched_skill") else "未命中可自动使用的 Skill",
            extra={
                "matched": bool(state.get("matched_skill")),
                "auto_use": bool(state.get("matched_skill")),
                "score": (state.get("matched_skill") or {}).get("match_score"),
            },
        )
        emit_visible_thought(self.db, state, "skill_matcher", stream_queue=self._stream_queue)
        return state

    def _format_recent_chat_messages_for_context(self, messages: list[Any]) -> str:
        """Format recent AgentChatMessage rows into [Conversation History] text."""
        if not messages:
            return ""
        lines: list[str] = []
        for msg in messages:
            role = getattr(msg, "role", "") or ""
            content = (getattr(msg, "content", "") or "").strip()
            if not content:
                continue
            if role == "user":
                label = "User"
            elif role == "assistant":
                label = "Assistant"
            else:
                label = role.capitalize() or "Message"
            max_len = 1200 if role == "assistant" else 600
            if len(content) > max_len:
                content = content[:max_len] + "..."
            lines.append(f"{label}: {content}")
        return "\n\n".join(lines)

    def _build_conversation_recall_context(
        self,
        messages: list[Any],
        *,
        current_user_input: str,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        previous_user_messages: list[str] = []
        current_normalized = self._normalize_recall_text(current_user_input)
        for msg in messages:
            role = getattr(msg, "role", "") or ""
            content = (getattr(msg, "content", "") or "").strip()
            if not content:
                continue
            rows.append({"role": role, "content": content})
            if role == "user" and self._normalize_recall_text(content) != current_normalized:
                previous_user_messages.append(content)
        return {
            "source": "AgentConversation/AgentMessage",
            "current_user_input": current_user_input,
            "messages": rows,
            "previous_user_messages": previous_user_messages,
            "message_count": len(rows),
            "previous_user_message_count": len(previous_user_messages),
        }

    @staticmethod
    def _normalize_recall_text(value: str) -> str:
        return "".join(str(value or "").split()).lower()

    @staticmethod
    def _dedupe_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or item.get("content") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _build_conversation_summary(self, run_id: int) -> str:
        """Build a short summary from recent agent steps of this run."""
        try:
            repo = AgentStepRepository(self.db)
            steps = repo.list_by_run(run_id)[-6:]
        except Exception:
            steps = []
        if not steps:
            return ""
        step_texts: list[str] = []
        for step in steps:
            node = getattr(step, "node_name", "") or ""
            status_val = getattr(step, "status", "") or ""
            output = getattr(step, "output", {}) or {}
            if isinstance(output, dict):
                route = output.get("route", "")
                if route:
                    step_texts.append(f"{node}(→{route}/{status_val})")
                else:
                    step_texts.append(f"{node}({status_val})")
            else:
                step_texts.append(node)
        return f"Agent Run {run_id} 已完成步骤：{' → '.join(step_texts)}。" if step_texts else ""

    def _build_checkpoint_summary(self, state: AgentRuntimeState) -> str:
        """Build a checkpoint summary from the current graph state."""
        parts = []
        route = state.get("route", "")
        if route:
            parts.append(f"当前路由：{route}")
        status = state.get("status", "")
        if status:
            parts.append(f"执行状态：{status}")
        artifacts = state.get("artifacts", [])
        if artifacts:
            parts.append(f"已生成 {len(artifacts)} 个 Artifact")
        skills = state.get("skill_drafts", []) or state.get("skill_drafts", [])
        if skills:
            parts.append(f"已创建 {len(skills)} 个 Skill 草稿")
        memory_count = len(state.get("memory_updates", []))
        if memory_count:
            parts.append(f"已写入 {memory_count} 条记忆")
        tool_call = state.get("tool_call")
        if tool_call:
            parts.append(f"工具调用：{tool_call.get('tool_name', '')} → {tool_call.get('status', '')}")
        return "；".join(parts) if parts else ""

    def _format_memories_for_context(self, memories: list[dict[str, Any]]) -> str:
        """Format memory dicts into readable Markdown text for ContextBuilder."""
        if not memories:
            return ""
        semantic = [m for m in memories if m.get("memory_type") == "semantic"]
        episodic = [m for m in memories if m.get("memory_type") == "episodic"]
        working = [m for m in memories if m.get("memory_type") == "working"]
        lines: list[str] = []

        def _meta_suffix(m: dict[str, Any]) -> str:
            parts = []
            score = m.get("_qdrant_score")
            if score:
                parts.append(f"score={score:.2f}")
            imp = m.get("importance")
            if imp:
                parts.append(f"importance={imp:.2f}")
            return f" ({', '.join(parts)})" if parts else ""

        if semantic:
            lines.append("## Semantic Memory (长期偏好/用户设定)")
            for m in semantic:
                lines.append(f"- {m.get('content', '')}{_meta_suffix(m)}")
        if episodic:
            lines.append("## Episodic Memory (历史任务/经验)")
            for m in episodic:
                lines.append(f"- {m.get('content', '')}{_meta_suffix(m)}")
        if working:
            lines.append("## Working Memory (当前任务临时状态)")
            for m in working:
                lines.append(f"- {m.get('content', '')}{_meta_suffix(m)}")
        return "\n".join(lines)

    def _format_profile_for_context(self, profile: Any) -> str:
        """Format user profile into readable text for ContextBuilder."""
        parts: list[str] = []
        segment = getattr(profile, "segment", "") or ""
        if segment:
            parts.append(f"segment: {segment}")
        goals = getattr(profile, "goals", "") or ""
        if goals:
            parts.append(f"goals: {goals}")
        interests = getattr(profile, "explicit_interests", "") or ""
        if interests:
            parts.append(f"interests: {interests}")
        return "\n".join(parts) if parts else ""

    def _format_rag_evidence_for_context(self, evidence: list[dict[str, Any]]) -> str:
        """Format RAG evidence list into readable text for ContextBuilder."""
        if not evidence:
            return ""
        lines: list[str] = ["## RAG Evidence (from user documents)"]
        for i, item in enumerate(evidence[:5], 1):
            source = item.get("source_name", "") or item.get("document_id", "")
            content = item.get("content", "")[:500]
            score = item.get("score", 0.0)
            lines.append(f"[{i}] score={score:.2f} | source={source}\n{content}")
        return "\n".join(lines)

    def _load_feed_card_context(self, user_id: int, feed_card_id: Any) -> dict[str, Any]:
        if not feed_card_id:
            return {}
        try:
            card = FeedRepository(self.db).get_by_user(user_id, int(feed_card_id))
        except (TypeError, ValueError):
            return {}
        if not card:
            return {}
        return {
            "id": card.id,
            "title": card.title,
            "one_sentence_value": card.one_sentence_value,
            "why_you": card.why_you,
            "information_gap": card.information_gap,
            "summary": card.one_sentence_value,
            "evidence": card.evidence,
            "suggested_actions": card.suggested_actions,
            "relation_type": card.exposure_bucket,
            "source_type": (card.score_detail or {}).get("source_type", ""),
            "domain": (card.score_detail or {}).get("domain", ""),
            "score": card.final_score,
        }

    def _skill_context_block(self, skill: dict[str, Any]) -> str:
        return "\n".join(
            [
                "Reusable Skill Applied:",
                f"- Skill Name: {skill.get('name', '')}",
                f"- Why matched: {skill.get('match_reason', '')}",
                f"- Expected Inputs: {skill.get('input_schema', {})}",
                f"- Execution Steps: {skill.get('tool_plan') or skill.get('context_recipe') or []}",
                f"- Output Contract: {skill.get('output_schema', {})}",
                f"- Constraints: safety_level={skill.get('safety_level', 'read_only')}",
            ]
        )

    # ── Memory write helpers ──────────────────────────────────────

    _MEMORY_WRITE_PATTERNS = [
        "记住", "帮我记", "记一下", "记下来", "记录一下",
        "以后记得", "别忘了", "下次记住",
        "以后都", "以后要", "以后都是", "以后都要",
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


