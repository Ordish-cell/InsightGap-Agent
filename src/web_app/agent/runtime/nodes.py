import json
import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.llm.config import get_llm_settings
from src.web_app.agent.llm.errors import LLMInvocationError, LLMParseError, LLMUnavailableError
from src.web_app.agent.llm.factory import get_chat_model
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.agent.llm.usage import record_llm_call
from src.web_app.agent.runtime.checkpoint import record_event, record_step
from src.web_app.agent.runtime.intent_llm import infer_home_intent_with_llm
from src.web_app.agent.runtime.intent_schema import HomeIntentResult
from src.web_app.agent.runtime.langgraph_status import append_status_step
from src.web_app.agent.runtime.planner import plan_route
from src.web_app.agent.runtime.router import route_user_input
from src.web_app.agent.runtime.state import AgentRuntimeState, append_error, append_output, mark_completed
from src.web_app.agent.runtime.visible_thoughts import emit_visible_thought, visible_thought_texts
from src.web_app.mcp.tool_router import infer_tool
from src.web_app.context.builder import ContextBuilder
from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository, AgentStepRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.artifact_repository import ArtifactRepository
from src.web_app.db.repositories.feed_repository import FeedRepository
from src.web_app.db.repositories.profile_repository import ProfileRepository
from src.web_app.research.schemas import ResearchRequest
from src.web_app.services.artifact_service import artifact_service
from src.web_app.services.memory_service import memory_service
from src.web_app.services.permission_service import PermissionGuard
from src.web_app.services.rag_service import rag_service
from src.web_app.services.research_service import research_service
from src.web_app.services.skill_service import skill_service
from src.web_app.services.mcp_service import mcp_service
from src.web_app.services.user_growth_service import user_growth_service

EXTERNAL_WRITE_TERMS = ("发邮件", "发送邮件", "邮件", "评论", "发布", "提交表单", "email", "send", "post", "submit")
HIGH_RISK_TERMS = ("删除", "支付", "付款", "转账", "delete", "payment")

logger = logging.getLogger(__name__)


class RuntimeNodes:
    def __init__(self, db: Session, payload: dict[str, Any]):
        self.db = db
        self.payload = payload

    async def permission_guard(self, state: AgentRuntimeState) -> AgentRuntimeState:
        text = state.get("user_input", "")
        permission_level = "L0_READ_ONLY"
        if any(term in text for term in HIGH_RISK_TERMS) or any(term in text for term in ("删除", "支付", "付款", "转账")):
            permission_level = L4_HIGH_RISK
        elif any(term in text for term in EXTERNAL_WRITE_TERMS) or any(term in text for term in ("发邮件", "发送邮件", "邮件", "评论", "发布", "提交表单")):
            permission_level = L3_EXTERNAL_WRITE

        decision = PermissionGuard().check_tool_call("agent_runtime_task", permission_level)
        state["permission"] = {"level": permission_level, **decision}
        if permission_level == L4_HIGH_RISK:
            state["permission"]["requires_approval"] = True
            state["permission"]["reason"] = "strong_approval_required"
        record_step(self.db, state["run_id"], "permission_guard", "permission", {"user_input": text}, {"permission": state["permission"], "route": state.get("route")})
        emit_visible_thought(self.db, state, "permission_guard")
        return state

    async def home_intent_react(self, state: AgentRuntimeState) -> AgentRuntimeState:
        user_input = state.get("user_input", "") or state.get("query", "")
        page_context = self.payload.get("page_context") or state.get("page_context") or {}
        feed_card_id = self.payload.get("feed_card_id") or page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
        thread_id = state.get("thread_id", "")
        record_event(
            self.db,
            state["run_id"],
            "home_intent_started",
            {"node_name": "home_intent_react", "input_preview": user_input[:200]},
            node_name="home_intent_react",
            user_id=state.get("user_id"),
            thread_id=thread_id,
        )
        home_intent = self._rule_home_intent(user_input, feed_card_id)
        fallback_reason = ""
        llm_settings = get_llm_settings()
        if llm_settings.enabled and llm_settings.intent_llm_enabled:
            try:
                llm_intent = infer_home_intent_with_llm(
                    self.db,
                    run_id=state["run_id"],
                    thread_id=thread_id,
                    user_id=state["user_id"],
                    user_input=user_input,
                    page_context=page_context,
                    selected_feed_card_id=feed_card_id,
                )
                home_intent = self._apply_rule_risk_floor(llm_intent, home_intent)
            except (LLMUnavailableError, LLMInvocationError, LLMParseError) as exc:
                fallback_reason = str(exc)
                home_intent["fallback_used"] = True
                home_intent["raw_intent_source"] = "fallback"
                record_event(
                    self.db,
                    state["run_id"],
                    "home_intent_fallback_used",
                    {"reason": fallback_reason[:200], "input_preview": user_input[:200]},
                    node_name="home_intent_react",
                    user_id=state.get("user_id"),
                    thread_id=thread_id,
                )
        state["home_intent"] = home_intent
        append_status_step(
            state,
            key="home_intent",
            node_name="home_intent_react",
            detail=f"识别为 {home_intent.get('intent') or home_intent.get('detected_intent')}，风险等级 {home_intent.get('risk_level')}",
            model=home_intent.get("model_used"),
            extra={
                "intent": home_intent.get("intent") or home_intent.get("detected_intent"),
                "risk_level": home_intent.get("risk_level"),
                "needs_approval": home_intent.get("needs_approval"),
                "fallback_used": home_intent.get("fallback_used", False),
                "reason_summary": home_intent.get("reason_summary") or home_intent.get("reasoning_summary", ""),
            },
        )
        emit_visible_thought(self.db, state, "home_intent_react")
        record_step(self.db, state["run_id"], "home_intent_react", "triage_intent", {"user_input": user_input, "page_context": page_context}, {"home_intent": home_intent})
        record_event(
            self.db,
            state["run_id"],
            "home_intent_completed",
            {"home_intent": home_intent, "fallback_reason": fallback_reason},
            node_name="home_intent_react",
            user_id=state.get("user_id"),
            thread_id=thread_id,
        )
        mark_completed(state, "home_intent_react")
        return state

    async def router(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        route = route_user_input(state.get("user_input", ""), self.payload)
        state["route"] = route
        record_step(self.db, state["run_id"], "router", "route", {"user_input": state.get("user_input", "")}, {"route": route})
        return state

    async def context_builder(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        route = state.get("route", "chat")
        user_input = state.get("user_input", "")
        profile = ProfileRepository(self.db).get_or_create_default(state["user_id"])
        memories = memory_service.search_memory(
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
        try:
            rag_evidence = rag_service.search_evidence(
                state["user_id"], user_input, limit=5, score_threshold=0.3,
            )
        except Exception:
            pass  # RAG failure must not block the pipeline

        # ── Format memories as readable text blocks ─────────────────
        memory_text = self._format_memories_for_context(memories)

        # ── Format profile as readable text ─────────────────────────
        profile_text = self._format_profile_for_context(profile)

        # ── Format RAG evidence as readable text ────────────────────
        evidence_text = self._format_rag_evidence_for_context(rag_evidence)

        builder = ContextBuilder(route=route)
        context_text, gssc_debug = builder.build_with_debug({
            "task": user_input,
            "route": route,
            "profile": profile_text,
            "memory": memory_text,
            "evidence": evidence_text,
            "feed_card": feed_card_context,
            "page_context": page_context,
            "conversation_history": conversation_history_text,
            "conversation_summary": conversation_summary,
            "checkpoint_summary": checkpoint_summary,
            "dynamic_preferences": dynamic_prefs.get("preference_summary", ""),
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
                     "recent_chat_messages_count": conversation_history_debug.get("recent_chat_messages_count", 0),
                     "has_conversation_history_text": conversation_history_debug.get("has_conversation_history_text", False),
                     "has_conversation_history_section": "Conversation History" in context_text,
                     "memory_search_backend": memory_search_backend,
                     "memory_qdrant_hits": memory_qdrant_hits,
                     **conversation_history_debug})
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
        emit_visible_thought(self.db, state, "context_builder")
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
        emit_visible_thought(self.db, state, "skill_matcher")
        return state

    async def research(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "research":
            return state
        request = ResearchRequest(
            query=self.payload.get("query") or state.get("user_input", ""),
            depth=self.payload.get("depth", "standard"),
            save_artifact=self.payload.get("save_artifact", True),
            write_memory=self.payload.get("write_memory", True),
            create_skill_draft=self.payload.get("create_skill_draft", True),
        )
        page_context = self.payload.get("page_context") or {}
        requested_feed_card_id = self.payload.get("feed_card_id") or page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
        loaded_feed_card = (state.get("context") or {}).get("feed_card") or {}
        try:
            requested_feed_card_id = int(requested_feed_card_id) if requested_feed_card_id else None
        except (TypeError, ValueError):
            requested_feed_card_id = None
        feed_card_id = loaded_feed_card.get("id") if requested_feed_card_id and loaded_feed_card.get("id") == requested_feed_card_id else None
        if feed_card_id:
            result = await research_service.research_feed_card(self.db, state["user_id"], int(feed_card_id), request)
        else:
            result = await research_service.research_query(self.db, state["user_id"], request)
        state["research"] = result
        state["final_output"] = result.get("summary") or result.get("status", "")
        state.setdefault("artifacts", [])
        if result.get("artifact_id"):
            state["artifacts"].append({"id": result["artifact_id"], "type": "research_report"})
        if result.get("skill_draft_id"):
            state.setdefault("skill_drafts", []).append({"id": result["skill_draft_id"], "source": "research"})
        record_step(self.db, state["run_id"], "research", "deep_research", {"feed_card_id": feed_card_id, "query": request.query}, {"research_run_id": result.get("id"), "status": result.get("status")})
        return state

    async def rag(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "rag":
            return state
        result = rag_service.ask(state["user_id"], state.get("user_input", ""), top_k=int(self.payload.get("top_k", 5)))
        state["rag"] = result
        state["final_output"] = result.get("answer", "")
        record_step(self.db, state["run_id"], "rag", "rag_ask", {"query": state.get("user_input", "")}, {"answer_mode": result.get("answer_mode"), "evidence_count": len(result.get("evidence", []))})
        return state

    async def artifact(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "artifact":
            return state
        content = state.get("final_output") or state.get("user_input", "")
        filename = f"agent_run_{state['run_id']}.md"
        file_path = artifact_service.save_text_artifact(state["user_id"], filename, content)
        item = ArtifactRepository(self.db).create(user_id=state["user_id"], run_id=state["run_id"], artifact_type="agent_output", title=f"Agent Output {state['run_id']}", file_path=file_path, metadata_json={"route": state.get("route")})
        artifact = {"id": item.id, "type": item.artifact_type, "file_path": item.file_path}
        state.setdefault("artifacts", []).append(artifact)
        state["final_output"] = f"Artifact saved: {item.title}"
        record_step(self.db, state["run_id"], "artifact", "save_artifact", {"filename": filename}, {"artifact": artifact})
        return state

    async def skill_librarian(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "skill":
            return state
        draft = skill_service.create_skill_draft_from_run(
            state["run_id"],
            user_id=state["user_id"],
            db=self.db,
            payload={
                "name": f"Agent runtime skill {state['run_id']}",
                "description": "Draft generated from a user-requested reusable workflow.",
                "trigger_text": state.get("user_input", ""),
                "tool_plan": ["permission_guard", "router", "context_builder", "runtime_node"],
                "eval_checks": ["deterministic_fallback", "no_external_write"],
            },
        )
        state.setdefault("skill_drafts", []).append(draft)
        state["final_output"] = f"Skill draft created: {draft['name']}"
        record_step(self.db, state["run_id"], "skill_librarian", "create_skill_draft", {"run_id": state["run_id"]}, {"skill_id": draft.get("id")})
        return state

    async def tool(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "tool":
            return state
        tool_name, tool_input = infer_tool(state.get("user_input", ""), self.payload)
        if not tool_name:
            state["status"] = "failed"
            state["error"] = "tool_not_found"
            state["final_output"] = "No MCP tool was selected."
            record_step(self.db, state["run_id"], "tool", "mcp_call", {"tool_name": None}, {"status": "failed", "error": state["error"]}, status="failed")
            return state
        result = mcp_service.call_tool(self.db, state["user_id"], tool_name, tool_input, agent_run_id=state["run_id"], dry_run=bool(self.payload.get("dry_run", False)))
        state["tool_call"] = result
        if result["status"] == "waiting_approval":
            state["status"] = "waiting_approval"
            state["final_output"] = f"MCP tool {tool_name} is waiting for approval."
        elif result["status"] in {"failed", "blocked"}:
            state["status"] = "failed"
            state["error"] = result.get("error", "")
            state["final_output"] = f"MCP tool {tool_name} failed: {state['error']}"
        else:
            state["final_output"] = f"MCP tool {tool_name} completed."
        record_step(self.db, state["run_id"], "tool", "mcp_call", {"tool_name": tool_name, "input": tool_input}, {"tool_call_id": result.get("id"), "status": result.get("status")}, status="completed" if result["status"] == "completed" else result["status"])
        return state

    async def memory_writer(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"} or state.get("status") == "waiting_approval":
            return state
        if not self.payload.get("write_memory", True):
            record_step(self.db, state["run_id"], "memory_writer", "write_memory", {"enabled": False}, {"reason": "disabled_by_request"})
            return state

        user_input = state.get("user_input", "")
        agent_output = state.get("final_output") or ""
        page_context = self.payload.get("page_context") or state.get("page_context") or {}
        feed_card_context = (state.get("context") or {}).get("feed_card") or {}
        matched_skill = state.get("matched_skill")
        created_skill_draft = state.get("created_skill_draft")

        try:
            result = memory_service.extract_and_save(
                user_id=state["user_id"],
                user_input=user_input,
                agent_output=agent_output,
                page_context=page_context,
                feed_card_context=feed_card_context,
                matched_skill=matched_skill,
                created_skill_draft=created_skill_draft,
                db=self.db,
            )
            saved = result.get("saved", {})
            all_saved = saved.get("working", []) + saved.get("episodic", []) + saved.get("semantic", [])
            state.setdefault("memory_updates", []).extend(all_saved)

            # Also write the basic episodic completion memory
            completion = memory_service.add_memory(
                user_id=state["user_id"],
                content=f"Agent runtime completed route={state.get('route')}: {agent_output[:200]}",
                memory_type="episodic",
                importance=0.6,
                metadata={"source_type": "agent_run", "source_id": str(state["run_id"]), "route": state.get("route")},
                db=self.db,
            )
            state["memory_updates"].append(completion)
            record_step(
                self.db, state["run_id"], "memory_writer", "write_memory",
                {"route": state.get("route"), "extraction": result.get("extraction", {}).get("should_consolidate", False)},
                {"memory_count": len(state["memory_updates"]), "semantic_count": len(saved.get("semantic", [])), "episodic_count": len(saved.get("episodic", []))},
            )
        except Exception:
            # Memory extraction failure must not break the agent run
            fallback = memory_service.add_memory(
                user_id=state["user_id"],
                content=f"Agent runtime completed route={state.get('route')}: {agent_output[:200]}",
                memory_type="episodic",
                importance=0.6,
                metadata={"source_type": "agent_run", "source_id": str(state["run_id"]), "route": state.get("route")},
                db=self.db,
            )
            state.setdefault("memory_updates", []).append(fallback)
            record_step(self.db, state["run_id"], "memory_writer", "write_memory", {"route": state.get("route"), "extraction_failed": True}, {"memory_id": fallback.get("id")})
        return state

    async def skill_draft_detector(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"} or state.get("status") == "waiting_approval":
            return state
        if self.payload.get("create_skill_draft_if_reusable", True) is False or self.payload.get("auto_skill", True) is False:
            record_step(self.db, state["run_id"], "skill_draft_detector", "detect_reuse", {"enabled": False}, {"created": False, "reason": "disabled_by_request"})
            return state
        reuse = skill_service.evaluate_reusability(state)
        state["skill_reuse"] = reuse
        created = None
        if reuse["should_create"] and not state.get("skill_drafts"):
            feed_card = (state.get("context") or {}).get("feed_card") or {}
            created = skill_service.create_skill_draft_from_run(
                state["run_id"],
                user_id=state["user_id"],
                db=self.db,
                payload={
                    "title": self._draft_title(state.get("user_input", "")),
                    "description": "Reusable workflow inferred from a completed Agent conversation.",
                    "trigger_patterns": [state.get("user_input", "")],
                    "input_schema": {"user_input": "string", "page_context": "object optional"},
                    "workflow_steps": ["permission_guard", "router", "context_builder", "skill_matcher", state.get("route", "runtime_node"), "memory_writer", "evaluator"],
                    "required_tools": [state.get("route", "agent_runtime")],
                    "output_contract": {"final_output": "string", "artifacts": "array", "memory_updates": "array"},
                    "safety_level": "read_only",
                    "eval_checks": ["no_external_write_without_approval", "reusable_score>=0.70"],
                    "source": self.payload.get("source", "agent_runtime"),
                    "source_agent_run_id": state["run_id"],
                    "source_feed_card_id": feed_card.get("id"),
                },
            )
            state.setdefault("skill_drafts", []).append(created)
            state["created_skill_draft"] = created
        record_step(self.db, state["run_id"], "skill_draft_detector", "detect_reuse", {"route": state.get("route")}, {"reusable_score": reuse["reusable_score"], "reason": reuse["reason"], "created_skill_draft": created})
        return state

    async def evaluator(self, state: AgentRuntimeState) -> AgentRuntimeState:
        is_resume = bool(state.get("_resume_context") or state.get("resolved_tool_call_ids"))
        logger.info(
            "[approval_resume_debug] node=evaluator before "
            "status=%s route=%s error=%s approval_required=%s is_resume=%s",
            state.get("status"), state.get("route"), state.get("error"),
            state.get("approval_required"), is_resume,
        )
        status = state.get("status") or ("failed" if state.get("error") else "completed")
        # Only set waiting_approval for the ORIGINAL pause flow, never on resume.
        if not is_resume and state.get("route") == "approval":
            status = "waiting_approval"
        state["status"] = status
        logger.info(
            "[approval_resume_debug] node=evaluator after new_status=%s", status,
        )
        state["evaluation"] = {
            "route": state.get("route"),
            "status": status,
            "has_output": bool(state.get("final_output")),
            "artifact_count": len(state.get("artifacts", [])),
            "memory_count": len(state.get("memory_updates", [])),
            "skill_count": len(state.get("skill_drafts", [])),
            "tool_call_id": (state.get("tool_call") or {}).get("id"),
        }
        if not state.get("final_output") and status == "completed":
            state["final_output"] = "Agent runtime completed."
        append_status_step(
            state,
            key="evaluator",
            node_name="evaluator",
            detail=f"评估完成，状态 {status}",
            extra={"status": status, "errors": len(state.get("errors", [])), "warnings": []},
        )
        emit_visible_thought(self.db, state, "evaluator")
        record_step(self.db, state["run_id"], "evaluator", "evaluate", {"route": state.get("route")}, {"evaluation": state["evaluation"]})
        mark_completed(state, "evaluator")
        return state

    # ── Multi-Agent Supervisor nodes ──────────────────────────────────

    async def planner(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Planner node: produce a RoutePlan from user input."""
        user_input = state.get("user_input", "") or state.get("query", "")
        feed_card_id = self.payload.get("feed_card_id")
        if not feed_card_id:
            page_context = self.payload.get("page_context") or state.get("page_context") or {}
            feed_card_id = page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")

        # Detect document attachments for routing
        _has_doc_attachments = bool(
            (self.payload.get("attachment_ids") or [])
            or (self.payload.get("attachment_context"))
            or (state.get("context") or {}).get("rag_evidence")
        )
        route_plan = plan_route(
            user_input=user_input,
            feed_card_id=feed_card_id,
            forced_route=self.payload.get("route"),
            forced_intent=self.payload.get("intent"),
            home_intent=state.get("home_intent"),
            has_document_attachments=_has_doc_attachments,
        )
        state["route_plan"] = route_plan
        route_intent = route_plan.get("intent", "chat")
        state["route"] = "tool" if str(route_intent).startswith("tool.") else route_intent  # legacy compat
        state["approval_required"] = route_plan.get("needs_approval", False)

        if route_plan.get("needs_approval"):
            state["approval_payload"] = {
                "risk_level": route_plan.get("risk_level"),
                "reason": route_plan.get("reason"),
                "user_input": user_input,
            }
            # Create approval record
            approval = ApprovalRepository(self.db).create(
                user_id=state["user_id"],
                run_id=state["run_id"],
                approval_type="agent_runtime",
                title=f"Approval required: {route_plan.get('risk_level')}",
                description=user_input,
                payload=state["approval_payload"],
            )
            state["approval_payload"]["approval_id"] = approval.id

        mark_completed(state, "planner")
        append_status_step(
            state,
            key="planner",
            node_name="planner",
            detail=f"计划调用 {len(route_plan.get('route', []))} 个关键节点，风险等级 {route_plan.get('risk_level', 'L0')}",
            model=resolve_model_name("planner").model,
            extra={
                "route": route_plan.get("route", []),
                "risk_level": route_plan.get("risk_level", "L0"),
                "steps_count": len(route_plan.get("route", [])),
            },
        )
        emit_visible_thought(self.db, state, "planner")
        record_step(self.db, state["run_id"], "planner", "plan_route",
                    {"user_input": user_input, "feed_card_id": feed_card_id},
                    {"route_plan": route_plan})
        record_event(
            self.db,
            state["run_id"],
            "plan_created",
            {"route_plan": route_plan, "requires_approval": state.get("approval_required", False)},
            node_name="planner",
            user_id=state.get("user_id"),
            thread_id=state.get("thread_id", ""),
        )
        return state

    async def research_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Research Agent: execute deep research via ResearchService."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "research_agent")
            return state
        try:
            request = ResearchRequest(
                query=self.payload.get("query") or state.get("user_input", ""),
                depth=self.payload.get("depth", "standard"),
                save_artifact=self.payload.get("save_artifact", True),
                write_memory=self.payload.get("write_memory", True),
                create_skill_draft=self.payload.get("create_skill_draft", True),
            )
            page_context = self.payload.get("page_context") or {}
            feed_card_id = (self.payload.get("feed_card_id")
                            or page_context.get("selected_feed_card_id")
                            or page_context.get("feed_card_id"))
            loaded_feed_card = (state.get("context") or {}).get("feed_card") or {}
            try:
                feed_card_id_int = int(feed_card_id) if feed_card_id else None
            except (TypeError, ValueError):
                feed_card_id_int = None
            if feed_card_id_int and loaded_feed_card.get("id") == feed_card_id_int:
                result = await research_service.research_feed_card(
                    self.db, state["user_id"], feed_card_id_int, request)
            else:
                result = await research_service.research_query(
                    self.db, state["user_id"], request)

            state["research"] = result
            state["research_result"] = result
            state["final_output"] = result.get("summary") or state.get("final_output", "")
            state.setdefault("artifacts", [])
            if result.get("artifact_id"):
                state["artifacts"].append({"id": result["artifact_id"], "type": "research_report"})
            if result.get("skill_draft_id"):
                state.setdefault("skill_drafts", []).append({"id": result["skill_draft_id"], "source": "research"})
            append_output(state, "research_agent", {"summary": result.get("summary", ""),
                          "findings": result.get("findings", []), "status": result.get("status")})
            record_step(self.db, state["run_id"], "research_agent", "deep_research",
                        {"feed_card_id": feed_card_id_int, "query": request.query},
                        {"research_run_id": result.get("id"), "status": result.get("status")})
            append_status_step(
                state,
                key="research_agent",
                node_name="research_agent",
                detail=f"研究状态 {result.get('status', 'completed')}，生成 Artifact {1 if result.get('artifact_id') else 0} 个",
                model=resolve_model_name("research", complexity="high").model,
                extra={
                    "summary": result.get("summary", ""),
                    "source_count": len(result.get("evidence", [])),
                    "artifact_count": 1 if result.get("artifact_id") else 0,
                },
            )
        except Exception as exc:
            append_error(state, "research_agent", str(exc))
            record_step(self.db, state["run_id"], "research_agent", "deep_research",
                        {}, {"error": str(exc)}, status="failed")
        emit_visible_thought(self.db, state, "research_agent")
        mark_completed(state, "research_agent")
        return state

    async def rag_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """RAG Agent: retrieve from user's knowledge base."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "rag_agent")
            return state
        try:
            route_plan = state.get("route_plan") or {}
            intent = route_plan.get("intent", "rag")
            user_input_str = state.get("user_input", "")
            # Determine if this is a document overview query
            from src.web_app.services.rag_service import is_document_overview_query
            overview = is_document_overview_query(user_input_str)
            # Get document IDs from context or payload
            page_context = state.get("page_context") or {}
            doc_ids_raw = (
                self.payload.get("attachment_ids")
                or page_context.get("attachment_ids")
                or None
            )
            if intent in ("document_qa",) and overview and doc_ids_raw:
                try:
                    doc_ids = [int(d) for d in doc_ids_raw]
                except (TypeError, ValueError):
                    doc_ids = None
                result = rag_service.ask_document(
                    state["user_id"],
                    user_input_str,
                    document_ids=doc_ids,
                    top_k=8,
                    overview_mode=True,
                )
            else:
                doc_ids_for_search = None
                if doc_ids_raw:
                    try:
                        doc_ids_for_search = [int(d) for d in doc_ids_raw]
                    except (TypeError, ValueError):
                        pass
                result = rag_service.ask(
                    state["user_id"], user_input_str,
                    top_k=int(self.payload.get("top_k", 5)),
                    document_ids=doc_ids_for_search,
                )
            state["rag"] = result
            state["rag_result"] = result
            append_output(state, "rag_agent", {"answer": result.get("answer", ""),
                          "evidence_count": len(result.get("evidence", []))})
            record_step(self.db, state["run_id"], "rag_agent", "rag_ask",
                        {"query": user_input_str, "intent": intent, "overview": overview},
                        {"answer_mode": result.get("answer_mode"),
                         "evidence_count": len(result.get("evidence", []))})
            append_status_step(
                state,
                key="rag_agent",
                node_name="rag_agent",
                detail=f"检索到 {len(result.get('evidence', []))} 条证据",
                model=resolve_model_name("rag").model,
                extra={
                    "evidence_count": len(result.get("evidence", [])),
                    "embedding_model": resolve_model_name("embedding").model,
                    "answer_model": resolve_model_name("rag").model,
                },
            )
        except Exception as exc:
            append_error(state, "rag_agent", str(exc))
            state["rag_result"] = {"answer": "", "evidence": [], "error": str(exc)}
        emit_visible_thought(self.db, state, "rag_agent")
        mark_completed(state, "rag_agent")
        return state

    async def artifact_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Artifact Agent: generate and save a document artifact."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "artifact_agent")
            return state
        try:
            # Build content from research/rag results or user_input
            content_parts = []
            research = state.get("research_result") or state.get("research") or {}
            if research.get("summary"):
                content_parts.append(f"# 研究摘要\n\n{research.get('summary', '')}")
            if research.get("findings"):
                content_parts.append("## 关键发现\n")
                for f in research.get("findings", [])[:5]:
                    content_parts.append(f"- {f}")
            rag = state.get("rag_result") or state.get("rag") or {}
            if rag.get("answer"):
                content_parts.append(f"# 知识库检索\n\n{rag.get('answer', '')}")
            if not content_parts:
                content_parts.append(f"# {state.get('user_input', 'Agent Output')}\n\n{state.get('final_output', '')}")
            content = "\n\n".join(content_parts)

            intent = (state.get("route_plan") or {}).get("intent", "chat")
            artifact_type_map = {
                "research": "research_report", "feed_research": "research_report",
                "artifact": "product_plan", "mixed": "structured_report",
            }
            artifact_type = artifact_type_map.get(intent, "markdown_report")

            filename = f"agent_run_{state['run_id']}_{artifact_type}.md"
            file_path = artifact_service.save_text_artifact(state["user_id"], filename, content)
            item = ArtifactRepository(self.db).create(
                user_id=state["user_id"], run_id=state["run_id"],
                artifact_type=artifact_type,
                title=f"Agent {artifact_type} {state['run_id']}",
                file_path=file_path,
                metadata_json={"route": state.get("route"), "intent": intent},
            )
            artifact = {"id": item.id, "type": item.artifact_type,
                        "title": item.title, "file_path": item.file_path}
            state.setdefault("artifacts", []).append(artifact)
            state["artifact_result"] = artifact
            append_output(state, "artifact_agent", artifact)
            record_step(self.db, state["run_id"], "artifact_agent", "save_artifact",
                        {"filename": filename}, {"artifact": artifact})
            append_status_step(
                state,
                key="artifact_agent",
                node_name="artifact_agent",
                detail=f"已生成 {artifact_type} Artifact",
                model=resolve_model_name("artifact").model,
                extra={"artifact_type": artifact_type, "artifact_id": item.id, "title": item.title},
            )
        except Exception as exc:
            append_error(state, "artifact_agent", str(exc))
        emit_visible_thought(self.db, state, "artifact_agent")
        mark_completed(state, "artifact_agent")
        return state

    async def tool_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """MCP Tool Agent: infer and execute tools with approval guard.

        On L3/L4 tools the executor returns waiting_approval WITHOUT executing.
        This node saves pause context (pending_*), sets status=waiting_approval,
        does NOT mark itself completed. The dispatcher detects waiting_approval
        and routes to END — a true graph interrupt.

        On resume (status=resuming), tool_agent checks resolved_tool_call_ids:
        if the previously-pending tool is now resolved, it accepts the pre-executed
        result, clears pending state, and marks itself completed so the dispatcher
        continues to evaluator → final_response.
        """
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "tool_agent")
            return state

        # Debug: tool_agent entry
        pending_tcid = state.get("pending_tool_call_id")
        resolved_ids_debug = list(state.get("resolved_tool_call_ids") or [])
        tc_debug = state.get("tool_call") or {}
        logger.info(
            "[approval_resume_debug] node=tool_agent status=%s approval_required=%s "
            "pending_tcid=%s resolved_ids=%s _resume_context=%s "
            "tool_call.error=%s tool_call.status=%s route=%s",
            state.get("status"), state.get("approval_required"),
            pending_tcid, resolved_ids_debug,
            state.get("_resume_context"),
            tc_debug.get("error") if isinstance(tc_debug, dict) else "N/A",
            tc_debug.get("status") if isinstance(tc_debug, dict) else "N/A",
            state.get("route"),
        )

        # ── Resume path: previously-pending tool was executed by resume runner ──
        pending_tcid = state.get("pending_tool_call_id")
        resolved_ids: list = list(state.get("resolved_tool_call_ids") or [])
        if pending_tcid is not None and pending_tcid in resolved_ids:
            # Accept the pre-executed tool result (already in state["tool_result"])
            state["approval_required"] = False
            state["approval_payload"] = None
            state["pending_approval_id"] = None
            state["pending_tool_name"] = None
            state["pending_tool_args"] = None
            state["pending_tool_call_id"] = None
            state["resume_token"] = None
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            return state

        # ── Resume path: tool failed after approval ───────────────
        # Detect tool failure: _resume_context starts with "failed:"
        resume_ctx = str(state.get("_resume_context") or "")
        if pending_tcid is not None and resume_ctx.startswith("failed:"):
            state["approval_required"] = False
            state["approval_payload"] = None
            state["pending_approval_id"] = None
            state["pending_tool_name"] = None
            state["pending_tool_args"] = None
            state["pending_tool_call_id"] = None
            state["resume_token"] = None
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            return state

        # ── Resume path: user rejected ────────────────────────────
        if pending_tcid is not None and resume_ctx.startswith("rejected:"):
            state["tool_call"] = {**state.get("tool_call", {}), "status": "rejected"}
            state["tool_result"] = {"status": "rejected", "message": "User rejected the approval"}
            state["approval_required"] = False
            state["approval_payload"] = None
            state["pending_approval_id"] = None
            state["pending_tool_name"] = None
            state["pending_tool_args"] = None
            state["pending_tool_call_id"] = None
            state["resume_token"] = None
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            return state

        # ── Resume path (legacy): status=resuming without context ──
        if pending_tcid is not None and state.get("status") == "resuming":
            state["approval_required"] = False
            state["approval_payload"] = None
            state["pending_approval_id"] = None
            state["pending_tool_name"] = None
            state["pending_tool_args"] = None
            state["pending_tool_call_id"] = None
            state["resume_token"] = None
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            return state

        try:
            route_plan = state.get("route_plan") or {}
            tool_name, tool_input = infer_tool(state.get("user_input", ""), {**self.payload, "intent": route_plan.get("intent")})
            if not tool_name:
                append_error(state, "tool_agent", "tool_not_found")
                mark_completed(state, "tool_agent")
                return state

            result = mcp_service.call_tool(self.db, state["user_id"], tool_name, tool_input,
                                           agent_run_id=state["run_id"],
                                           dry_run=bool(self.payload.get("dry_run", False)))
            state["tool_call"] = result
            state["tool_result"] = result

            if result["status"] == "waiting_approval":
                state["status"] = "waiting_approval"
                state["approval_required"] = True
                approval_id = result.get("approval_id") or (result.get("output", {}).get("_metadata", {}).get("approval_id"))
                tool_call_id = result.get("id")
                # ── Save pause context for resume ─────────────────
                # pending_tool_args stores the REAL args (for resume execution).
                # approval_payload gets sanitized args (for frontend).
                state["pending_approval_id"] = str(approval_id) if approval_id else None
                state["pending_tool_name"] = tool_name
                state["pending_tool_args"] = dict(tool_input)
                state["pending_tool_call_id"] = tool_call_id
                state["route_plan_snapshot"] = dict(route_plan)
                state["resume_token"] = f"approval:{approval_id}"
                state["approval_payload"] = {
                    "approval_id": approval_id,
                    "tool_name": tool_name,
                    "risk_level": route_plan.get("risk_level", "L3"),
                    "tool_args": _sanitize_tool_args(tool_name, tool_input),
                    "preview": result.get("output", {}),
                    "run_id": state["run_id"],
                    "user_id": state["user_id"],
                    "title": f"需要你确认：{tool_name}",
                    "actions": ["approve", "reject"],
                }
                append_output(state, "tool_agent", {"tool_name": tool_name, "status": "waiting_approval"})
                record_step(self.db, state["run_id"], "tool_agent", "mcp_approval_required",
                            {"tool_name": tool_name, "tool_input": _sanitize_tool_args(tool_name, tool_input)},
                            {"status": "waiting_approval", "approval_id": approval_id})
                append_status_step(
                    state,
                    key="tool_agent",
                    node_name="tool_agent",
                    status="waiting_approval",
                    detail=f"工具动作需要审批，风险等级 {route_plan.get('risk_level', 'L3')}",
                    extra={"risk_level": route_plan.get("risk_level", "L3"), "tool_name": tool_name, "approval_required": True},
                )
                emit_visible_thought(self.db, state, "tool_agent")
                # NOT marked completed → dispatcher routes to END (true interrupt)
                return state
            elif result["status"] in {"failed", "blocked"}:
                state["status"] = "failed"
                state["error"] = result.get("error", "")
                state["final_output"] = f"工具 {tool_name} 失败: {result.get('error', 'unknown')}"
                append_error(state, "tool_agent", result.get("error", "tool_failed"))
                record_step(self.db, state["run_id"], "tool_agent", "mcp_call",
                            {"tool_name": tool_name}, {"status": result["status"], "error": result.get("error")}, status="failed")
                append_status_step(
                    state,
                    key="tool_agent",
                    node_name="tool_agent",
                    detail=f"工具 {tool_name} 执行失败",
                    extra={"risk_level": route_plan.get("risk_level", "L0"), "status": result["status"]},
                )
            else:
                append_output(state, "tool_agent", {"tool_name": tool_name, "status": result.get("status")})
                record_step(self.db, state["run_id"], "tool_agent", "mcp_call",
                            {"tool_name": tool_name}, {"status": result.get("status")})
                append_status_step(
                    state,
                    key="tool_agent",
                    node_name="tool_agent",
                    detail=f"工具 {tool_name} 状态 {result.get('status')}",
                    extra={"risk_level": route_plan.get("risk_level", "L0"), "dry_run": bool(self.payload.get("dry_run", False)), "approval_required": False},
                )
        except Exception as exc:
            append_error(state, "tool_agent", str(exc))
        emit_visible_thought(self.db, state, "tool_agent")
        mark_completed(state, "tool_agent")
        return state

    async def memory_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Memory Agent: write semantic memories for explicit user requests,
        and conditionally extract memories for high-value tasks."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "memory_agent")
            return state
        try:
            route_plan = state.get("route_plan") or {}
            intent = route_plan.get("intent", "chat")
            user_input = state.get("user_input", "")

            # ── Explicit memory write: user says "记住" / "帮我记" etc. ──
            if self._is_explicit_memory_write(user_input):
                memory_content = self._extract_memory_from_user_input(user_input)
                mem = memory_service.add_memory(
                    user_id=state["user_id"],
                    content=memory_content,
                    memory_type="semantic",
                    importance=0.9,
                    metadata={"source_type": "explicit_user_request", "source": "explicit_user_request",
                              "raw_user_input": user_input, "explicit": True, "run_id": str(state["run_id"])},
                    db=self.db,
                )
                state.setdefault("memory_updates", []).append(mem)
                state["memory_write_result"] = {
                    "success": True,
                    "content": memory_content,
                    "memory_type": "semantic",
                    "memory_id": mem.get("id"),
                }
                state["memory_result"] = {"saved_count": 1, "semantic": 1, "episodic": 0}
                state["final_output"] = f"已记住：{memory_content}"
                append_output(state, "memory_agent", state["memory_write_result"])
                record_step(self.db, state["run_id"], "memory_agent", "write_memory_explicit",
                            {"user_input": user_input},
                            {"memory_write_result": state["memory_write_result"]})
                append_status_step(
                    state,
                    key="memory_agent",
                    node_name="memory_agent",
                    detail=f"已明确写入 semantic 记忆 1 条",
                    model=resolve_model_name("memory", complexity="low").model,
                    extra={"memory_writes": 1, "explicit": True},
                )
                emit_visible_thought(self.db, state, "memory_agent")
                mark_completed(state, "memory_agent")
                return state

            # Only write memory for high-value tasks, not casual chat
            if intent in ("chat",):
                mark_completed(state, "memory_agent")
                return state

            # Aggregate output from all agent results for memory extraction
            research = state.get("research_result") or state.get("research") or {}
            rag = state.get("rag_result") or state.get("rag") or {}
            agent_output = (
                state.get("final_output")
                or research.get("summary", "")
                or rag.get("answer", "")
                or ""
            )
            page_context = self.payload.get("page_context") or state.get("page_context") or {}
            feed_card_context = (state.get("context") or {}).get("feed_card") or {}
            matched_skill = state.get("matched_skill")
            created_skill_draft = state.get("created_skill_draft")

            result = memory_service.extract_and_save(
                user_id=state["user_id"], user_input=user_input,
                agent_output=agent_output, page_context=page_context,
                feed_card_context=feed_card_context,
                matched_skill=matched_skill,
                created_skill_draft=created_skill_draft,
                db=self.db,
            )
            saved = result.get("saved", {})
            all_saved = saved.get("working", []) + saved.get("episodic", []) + saved.get("semantic", [])
            state.setdefault("memory_updates", []).extend(all_saved)
            state["memory_result"] = {"saved_count": len(all_saved),
                                       "semantic": len(saved.get("semantic", [])),
                                       "episodic": len(saved.get("episodic", []))}
            append_output(state, "memory_agent", state["memory_result"])
            record_step(self.db, state["run_id"], "memory_agent", "write_memory", {},
                        {"memory_result": state["memory_result"]})
            append_status_step(
                state,
                key="memory_agent",
                node_name="memory_agent",
                detail=f"写入记忆 {state['memory_result']['saved_count']} 条",
                model=resolve_model_name("memory", complexity="low").model,
                extra={"memory_writes": state["memory_result"]["saved_count"]},
            )
        except Exception as exc:
            append_error(state, "memory_agent", str(exc))
        emit_visible_thought(self.db, state, "memory_agent")
        mark_completed(state, "memory_agent")
        return state

    async def skill_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Skill Agent: detect reusable workflows and optionally create skill drafts."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "skill_agent")
            return state
        try:
            reuse = skill_service.evaluate_reusability(state)
            state["skill_reuse"] = reuse
            created = None
            if reuse.get("should_create") and not state.get("skill_drafts"):
                feed_card = (state.get("context") or {}).get("feed_card") or {}
                created = skill_service.create_skill_draft_from_run(
                    state["run_id"], user_id=state["user_id"], db=self.db,
                    payload={
                        "title": self._draft_title(state.get("user_input", "")),
                        "description": "Reusable workflow from multi-agent run.",
                        "trigger_patterns": [state.get("user_input", "")],
                        "input_schema": {"user_input": "string", "page_context": "object optional"},
                        "workflow_steps": (state.get("route_plan") or {}).get("route", []),
                        "required_tools": [(state.get("route_plan") or {}).get("intent", "chat")],
                        "output_contract": {"final_output": "string", "artifacts": "array"},
                        "safety_level": (state.get("route_plan") or {}).get("risk_level", "L0"),
                        "eval_checks": ["no_external_write_without_approval"],
                        "source": self.payload.get("source", "agent_runtime"),
                        "source_agent_run_id": state["run_id"],
                        "source_feed_card_id": feed_card.get("id"),
                    },
                )
                state.setdefault("skill_drafts", []).append(created)
                state["created_skill_draft"] = created
            state["skill_result"] = {
                "reusable_score": reuse.get("reusable_score", 0),
                "should_create": reuse.get("should_create", False),
                "created": created is not None,
                "reason": reuse.get("reason", ""),
            }
            append_output(state, "skill_agent", state["skill_result"])
            record_step(self.db, state["run_id"], "skill_agent", "detect_reuse", {},
                        {"skill_result": state["skill_result"], "created_skill_draft": created})
            append_status_step(
                state,
                key="skill_agent",
                node_name="skill_agent",
                detail="已生成 Skill 草稿" if created else "未生成 Skill 草稿",
                model=resolve_model_name("skill", complexity="low").model,
                extra={"skill_drafts": len(state.get("skill_drafts", [])), "created": created is not None},
            )
        except Exception as exc:
            append_error(state, "skill_agent", str(exc))
        emit_visible_thought(self.db, state, "skill_agent")
        mark_completed(state, "skill_agent")
        return state

    async def final_response(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Final Response: use the configured final LLM to write the user-facing answer.

        Only reached when all route nodes have completed (or on reject/resume).
        Never reached during waiting_approval — the graph interrupts to END before
        reaching this node.
        """
        route_plan = state.get("route_plan") or {}
        intent = route_plan.get("intent", "chat")

        logger.info(
            "[approval_resume_debug] node=final_response before "
            "status=%s error=%s approval_required=%s "
            "_resume_context=%s tool_result.status=%s",
            state.get("status"), state.get("error"),
            state.get("approval_required"),
            state.get("_resume_context"),
            (state.get("tool_result") or {}).get("status", "N/A"),
        )

        answer_parts = []
        research = state.get("research_result") or state.get("research") or {}
        if research.get("summary") and intent not in ("document_qa",):
            answer_parts.append(research["summary"])
        rag = state.get("rag_result") or state.get("rag") or {}
        rag_answer = rag.get("answer", "")
        if rag_answer and rag_answer != "[document_qa_context]":
            answer_parts.append(rag_answer)
        if state.get("final_output") and not answer_parts:
            answer_parts.append(state["final_output"])

        # ── Enrich visible thoughts BEFORE generating the final answer ──
        if intent != "chat":
            emit_visible_thought(self.db, state, "final_response")
            await self._enrich_visible_thoughts_with_llm(state)

        draft_answer = "\n\n".join(answer_parts)
        used_streaming_llm = False
        if intent in ("document_qa",):
            # Document Q&A: always use LLM to rewrite, never echo raw chunks
            final_answer = await self._generate_final_answer_with_llm(state, draft_answer)
            used_streaming_llm = True
        elif draft_answer.strip() and not self._is_generic_draft_answer(draft_answer):
            final_answer = draft_answer.strip()
        else:
            final_answer = await self._generate_final_answer_with_llm(state, draft_answer)
            used_streaming_llm = True

        # ── Ensure streaming flags survive LangGraph state serialisation ──
        # _generate_final_answer_with_llm sets these inside the astream loop,
        # but LangGraph may copy state between nodes. Re-assert here so the
        # fallback guard in agent_service sees them.
        if used_streaming_llm:
            state["_answer_delta_emitted"] = True
            state["_answer_completed_emitted"] = True

        state["final_answer"] = final_answer
        state["final_output"] = final_answer

        errors = state.get("errors", [])
        status = state.get("status") or ("failed" if errors else "completed")

        final_payload = {
            "run_id": str(state.get("run_id", "")),
            "thread_id": state.get("thread_id", ""),
            "intent": intent,
            "route": route_plan.get("route", []),
            "answer": final_answer,
            "cards": [],
            "research": research,
            "rag": rag,
            "artifacts": state.get("artifacts", []),
            "tool_calls": [state.get("tool_call")] if state.get("tool_call") else [],
            "approval_required": state.get("approval_required", False),
            "approval_payload": state.get("approval_payload"),
            "memory_writes": state.get("memory_updates", []),
            "skill_drafts": state.get("skill_drafts", []),
            "evaluation": state.get("evaluation", {}),
            "errors": errors,
            "agent_outputs": state.get("agent_outputs", []),
        }
        append_status_step(
            state,
            key="final_response",
            node_name="final_response",
            detail="已生成最终回答" if final_answer else "最终回答为空",
            model=resolve_model_name("final").model,
            extra={
                "answer_generated": bool(final_answer),
                "cards_count": 0,
                "artifacts_count": len(state.get("artifacts", [])),
                "tool_calls_count": 1 if state.get("tool_call") else 0,
            },
        )
        # Chat fast-path: no user-visible progress. Research/artifact/tool
        # intents already emitted milestones before answer generation (above).
        final_payload["thinking_summary"] = visible_thought_texts(state)
        final_payload["visible_thoughts"] = state.get("visible_thoughts", [])
        final_payload["langgraphstatus"] = state.get("langgraphstatus", {})
        state["final_payload"] = final_payload
        state["status"] = status

        gssc_context = (state.get("context") or {}).get("gssc_context", "")
        gssc_debug = (state.get("context") or {}).get("gssc_debug", {})
        record_step(self.db, state["run_id"], "final_response", "aggregate",
                    {"intent": intent},
                    {"status": status, "answer_len": len(final_answer),
                     "artifact_count": len(state.get("artifacts", [])),
                     "error_count": len(errors),
                     "final_prompt_uses_gssc_context": bool(gssc_context),
                     "gssc_context_chars": len(gssc_context),
                     "gssc_context_tokens_estimate": max(1, len(gssc_context) // 4),
                     "has_memory_section": "Relevant Memory" in gssc_context or "[Relevant Memory]" in gssc_context,
                     "has_rag_evidence_section": "Evidence" in gssc_context or "[Evidence]" in gssc_context,
                     "has_profile_section": "User Profile" in gssc_context or "[User Profile]" in gssc_context,
                     "has_conversation_history_section": "Conversation History" in gssc_context or "[Conversation History]" in gssc_context,
                     "gssc_selected_sources": gssc_debug.get("selected_sources", []),
                     "gssc_dropped_sources": gssc_debug.get("dropped_sources", []),
                     })
        logger.info(
            "[approval_resume_debug] node=final_response after "
            "status=%s final_answer_preview=%s",
            state.get("status"), (state.get("final_answer") or "")[:120],
        )
        return state

    async def _enrich_visible_thoughts_with_llm(self, state: AgentRuntimeState) -> None:
        """Optionally polish visible stage summaries without exposing private chain-of-thought."""
        thoughts = list(state.get("visible_thoughts") or [])
        if not thoughts or not get_llm_settings().enabled:
            return

        resolution = resolve_model_name("final")
        prompt = self._build_visible_thought_prompt(state, thoughts)
        started = time.perf_counter()
        output_text = ""
        try:
            model = get_chat_model("final", temperature=0.2)
            message = await model.ainvoke(prompt)
            output_text = self._message_content(message).strip()
            rows = self._parse_trace_rows(output_text)
            if not rows:
                return
            by_key = {str(item.get("key") or ""): item for item in rows if item.get("key")}
            enriched_thoughts = []
            for thought in thoughts:
                key = str(thought.get("key") or "")
                row = by_key.get(key)
                if row:
                    thought = {**thought, "text": str(row.get("text") or row.get("summary") or thought.get("text") or "")}
                enriched_thoughts.append(thought)
            state["visible_thoughts"] = enriched_thoughts
            status = state.setdefault("langgraphstatus", {})
            status["visible_thoughts"] = enriched_thoughts
            status["trace_style"] = "visible_thought_summary"
            state["langgraphstatus"] = status
            latency_ms = int((time.perf_counter() - started) * 1000)
            record_llm_call(
                self.db,
                run_id=state.get("run_id"),
                thread_id=state.get("thread_id", ""),
                user_id=state.get("user_id"),
                node_name="trace_visualizer",
                purpose="final",
                provider=resolution.provider,
                model=resolution.model,
                tier=resolution.tier,
                latency_ms=latency_ms,
                status="completed",
                estimated_input_chars=len(prompt),
                estimated_output_chars=len(output_text),
                metadata={"stage_count": len(enriched_thoughts)},
            )
            record_event(
                self.db,
                state["run_id"],
                "thought_summary",
                {"title": "trace_visualizer", "summary": "Generated visible stage summaries.", "stage_count": len(enriched_thoughts), "model": resolution.model},
                node_name="trace_visualizer",
                user_id=state.get("user_id"),
                thread_id=state.get("thread_id", ""),
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            record_llm_call(
                self.db,
                run_id=state.get("run_id"),
                thread_id=state.get("thread_id", ""),
                user_id=state.get("user_id"),
                node_name="trace_visualizer",
                purpose="final",
                provider=resolution.provider,
                model=resolution.model,
                tier=resolution.tier,
                latency_ms=latency_ms,
                status="failed",
                error_message=str(exc),
                estimated_input_chars=len(prompt),
                estimated_output_chars=len(output_text),
                metadata={"stage_count": len(thoughts)},
            )

    def _build_visible_thought_prompt(self, state: AgentRuntimeState, thoughts: list[dict[str, Any]]) -> str:
        rows = [
            {
                "key": item.get("key"),
                "status": item.get("status"),
                "text": item.get("text"),
            }
            for item in thoughts
        ]
        payload = {
            "user_input": state.get("user_input", ""),
            "intent": (state.get("route_plan") or {}).get("intent") or (state.get("home_intent") or {}).get("intent", "chat"),
            "risk_level": (state.get("route_plan") or {}).get("risk_level") or (state.get("home_intent") or {}).get("risk_level", "L0"),
            "visible_thoughts": rows,
        }
        return (
            "You polish visible progress narration for an agent UI, similar to Codex status updates.\n"
            "Do not reveal hidden chain-of-thought. Do not mention internal node names or ReAct fields.\n"
            "Return strict JSON only: an array of objects with keys: key, text.\n"
            "Use Simplified Chinese. Each text value must be one natural user-facing sentence.\n"
            "Explain what is happening, why it matters, or what will happen next.\n"
            f"Input data: {json.dumps(payload, ensure_ascii=False, default=str)}"
        )

    def _parse_trace_rows(self, text: str) -> list[dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    async def _generate_final_answer_with_llm(self, state: AgentRuntimeState, draft_answer: str) -> str:
        user_input = state.get("user_input", "")
        if not get_llm_settings().enabled:
            return self._fallback_final_answer(state, draft_answer)

        from src.web_app.agent.runtime.events import queue_stream_event  # noqa: F811

        resolution = resolve_model_name("final")
        prompt = self._build_final_answer_prompt(state, draft_answer)
        started = time.perf_counter()
        full_answer = ""
        run_id = state.get("run_id")
        thread_id = state.get("thread_id", "")
        user_id = state.get("user_id")
        queue = state.get("_stream_queue")
        try:
            model = get_chat_model("final", temperature=0.35, streaming=True)
            # Emit answer_started (SSE + DB)
            if queue:
                queue_stream_event(queue, "answer_started", {}, run_id=run_id, thread_id=thread_id, node_name="final_response")
            record_event(
                self.db, run_id, "answer_started", {},
                node_name="final_response", user_id=user_id, thread_id=thread_id,
            )
            state["_answer_started_emitted"] = True

            chunk_index = 0
            async for chunk in model.astream(prompt):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if not content:
                    continue
                full_answer += content
                chunk_index += 1
                # Push to SSE only — do NOT persist each token to agent_events
                if queue:
                    queue_stream_event(
                        queue, "answer_delta",
                        {"text": content, "index": chunk_index},
                        run_id=run_id, thread_id=thread_id, node_name="final_response",
                    )

            full_answer = full_answer.strip()
            # ── Guard: detect if LLM output internal JSON despite prompt ──
            if self._looks_like_internal_json(full_answer):
                extracted = self._extract_text_from_json_output(full_answer)
                if extracted:
                    # Replace the streamed JSON with the extracted text
                    if queue:
                        queue_stream_event(
                            queue, "answer_completed",
                            {"answer": extracted, "status": "corrected"},
                            run_id=run_id, thread_id=thread_id, node_name="final_response",
                        )
                    record_event(
                        self.db, run_id, "answer_json_corrected",
                        {"original_len": len(full_answer), "extracted_len": len(extracted)},
                        node_name="final_response", user_id=user_id, thread_id=thread_id,
                    )
                    full_answer = extracted
            if not full_answer:
                raise LLMInvocationError("Final LLM returned empty output")

            latency_ms = int((time.perf_counter() - started) * 1000)
            record_llm_call(
                self.db,
                run_id=run_id, thread_id=thread_id, user_id=user_id,
                node_name="final_response", purpose="final",
                provider=resolution.provider, model=resolution.model, tier=resolution.tier,
                latency_ms=latency_ms, status="completed",
                estimated_input_chars=len(prompt),
                estimated_output_chars=len(full_answer),
                metadata={"input_preview": user_input[:200], "streaming": True, "chunks": chunk_index},
            )
            record_event(
                self.db, run_id, "thought_summary",
                {"title": "生成最终回答", "summary": "已流式调用最终回复模型，把执行结果整理成用户可读回答。", "model": resolution.model},
                node_name="final_response", user_id=user_id, thread_id=thread_id,
            )
            # Emit answer_completed (SSE + DB)
            if queue:
                queue_stream_event(
                    queue, "answer_completed", {"answer": full_answer},
                    run_id=run_id, thread_id=thread_id, node_name="final_response",
                )
            record_event(
                self.db, run_id, "answer_completed", {"answer": full_answer},
                node_name="final_response", user_id=user_id, thread_id=thread_id,
            )
            # Mark that streaming happened so agent_service fallback is skipped
            state["_answer_delta_emitted"] = True
            state["_answer_completed_emitted"] = True
            return full_answer

        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            record_llm_call(
                self.db,
                run_id=run_id, thread_id=thread_id, user_id=user_id,
                node_name="final_response", purpose="final",
                provider=resolution.provider, model=resolution.model, tier=resolution.tier,
                latency_ms=latency_ms, status="failed", error_message=str(exc),
                estimated_input_chars=len(prompt),
                estimated_output_chars=len(full_answer),
                metadata={"input_preview": user_input[:200], "streaming": True},
            )
            # Emit partial answer_completed on error so SSE is not left hanging
            if queue:
                queue_stream_event(
                    queue, "answer_completed",
                    {"answer": full_answer, "status": "partial", "error": str(exc)},
                    run_id=run_id, thread_id=thread_id, node_name="final_response",
                )
            record_event(
                self.db, run_id, "answer_completed",
                {"answer": full_answer, "status": "partial", "error": str(exc)},
                node_name="final_response", user_id=user_id, thread_id=thread_id,
            )
            record_event(
                self.db, run_id, "thought_summary",
                {"title": "最终模型不可用", "summary": "最终回复模型调用失败，已改用安全兜底回答。", "error": str(exc)[:200]},
                node_name="final_response", user_id=user_id, thread_id=thread_id,
            )
            # Still mark as emitted so agent_service fallback doesn't double-push
            state["_answer_delta_emitted"] = True
            state["_answer_completed_emitted"] = True
            return self._fallback_final_answer(state, draft_answer)

    def _build_final_answer_prompt(self, state: AgentRuntimeState, draft_answer: str) -> str:
        route_plan = state.get("route_plan") or {}
        context = state.get("context") or {}
        gssc_context = context.get("gssc_context", "")

        # ── Build the core prompt ───────────────────────────────────
        # When GSSC context is available, it becomes the primary context.
        # When it's empty, fall back to a legacy flat payload.
        if gssc_context:
            return self._build_gssc_prompt(state, gssc_context, draft_answer, route_plan)
        return self._build_legacy_prompt(state, draft_answer, route_plan)

    def _build_gssc_prompt(
        self,
        state: AgentRuntimeState,
        gssc_context: str,
        draft_answer: str,
        route_plan: dict[str, Any],
    ) -> str:
        """Build the final LLM prompt with GSSC as the primary context source."""
        rag_result = state.get("rag_result") or state.get("rag") or {}
        research_result = state.get("research_result") or state.get("research") or {}
        artifacts = state.get("artifacts", [])
        tool_result = state.get("tool_result") or state.get("tool_call") or {}
        errors = state.get("errors", [])

        system_instruction = (
            "你是信息差 Agent OS 的最终回复节点。你必须基于下面的结构化上下文，用自然语言回答用户。\n\n"

            f"[Structured GSSC Context]\n{gssc_context}\n\n"

            f"[Current User Input]\n{state.get('user_input', '')}\n\n"
        )

        # Append specialized agent results only when they carry new information
        # not already covered by GSSC (rag_agent runs AFTER context_builder).
        extra_blocks: list[str] = []
        attachment_context = (
            (state.get("context") or {}).get("page_context", {}).get("attachment_context", "")
            or (state.get("page_context") or {}).get("attachment_context", "")
        )
        if attachment_context:
            extra_blocks.append(
                "[用户上传附件上下文]\n"
                "以下内容来自用户在当前对话中上传的图片或文件。\n"
                "这些内容应作为回答当前问题的主要依据。\n"
                "如果这里包含图片理解结果，请直接基于图片理解结果回答。\n"
                "不要仅因为外部搜索证据不足就拒绝回答。\n\n"
                f"{attachment_context}"
            )
        if rag_result.get("answer"):
            rag_answer_text = rag_result.get("answer", "")
            rag_ctx = rag_result.get("context") or {}
            doc_block = rag_ctx.get("document_context_block", "")
            if doc_block and rag_answer_text == "[document_qa_context]":
                # Document Q&A mode: inject structured document context directly
                extra_blocks.append(doc_block)
            elif doc_block and rag_answer_text != "[document_qa_context]":
                extra_blocks.append(doc_block)
                extra_blocks.append(
                    f"[RAG Agent Result]\n{rag_answer_text}\n"
                    f"Evidence count: {len(rag_result.get('evidence', []))}"
                )
            else:
                extra_blocks.append(
                    f"[RAG Agent Result]\n{rag_answer_text}\n"
                    f"Evidence count: {len(rag_result.get('evidence', []))}"
                )
        if research_result.get("summary"):
            extra_blocks.append(
                f"[Research Agent Result]\n{research_result.get('summary', '')}"
            )
        if artifacts:
            extra_blocks.append(
                f"[Artifacts]\n" +
                "\n".join(a.get("title", a.get("id", "")) for a in artifacts[:5])
            )
        if tool_result.get("status"):
            extra_blocks.append(
                f"[Tool Result]\n"
                f"Tool: {tool_result.get('tool_name', '')}\n"
                f"Status: {tool_result.get('status', '')}"
            )
        if errors:
            extra_blocks.append(
                f"[Errors]\n" + "\n".join(e.get("error", str(e)) for e in errors[:3])
            )
        if extra_blocks:
            system_instruction += "\n".join(extra_blocks) + "\n\n"

        intent = route_plan.get("intent", "chat")
        if intent in ("document_qa",) and ("rag" in str(route_plan.get("route", []))):
            system_instruction += (
                "[Document Q&A Instructions]\n"
                "你正在回答用户关于当前上传文档的问题。\n"
                "请根据以下上下文直接回答用户。\n"
                "1. 使用自然中文，像正常助手一样回答，自然地组织为段落。\n"
                "2. 严禁出现以下内部术语：evidence item、research question、information-gap opportunity、"
                "RAG、chunk、知识库证据、Based on X evidence items、Evidence is insufficient、"
                "当前知识库中没有找到足够证据、research question、limited to available context。\n"
                "3. 如果用户问「文档里讲了啥/总结一下」，请输出：\n"
                "   - 一句话总结\n   - 主要内容 3-5 条，用 Markdown 标题和无序列表\n"
                "   - 如果解析内容有限，最后说明限制\n"
                "4. 如果文档是项目计划/课程报告/实验报告，请主动识别文档类型。\n"
                "5. 不要编造没有出现在文档中的内容。\n"
                "6. 如果当前上下文确实没有文档内容（例如文档解析失败/为空），请用中文友好地告知用户。\n"
                "7. 输出简洁、可读、像正常助手回答。\n"
                "\n"
            )
        system_instruction += (
            "[Instructions]\n"
            "1. 优先使用 Structured GSSC Context 中的 Memory、Profile、Evidence、Feed、Conversation 信息。\n"
            "2. 如果 Memory 或 Evidence 中没有相关信息，不要编造。\n"
            "3. 如果用户问「我之前说过什么」「我的偏好是什么」「我们聊过什么」，必须优先从 GSSC Context 的 Memory 和 Conversation 部分回答。\n"
        )
        system_instruction += (
            "4. 如果用户询问「我问过你什么 / 我刚才问过什么 / 我之前问过什么 / 是否问过某主题 / 刚才聊了什么」，必须优先查看 [Conversation History]。\n"
            "5. 判断用户是否问过某主题时，只依据 [Conversation History] 中的 User 消息，不要依据 Assistant 的旧回答。\n"
            "6. 如果 [Conversation History] 中存在相关 User 消息，请直接列出这些用户问题。\n"
            "7. 当前会话历史判断优先级高于 Memory、RAG Evidence、Feed Card 和 Dynamic Preferences。\n"
        )
        if intent in ("document_qa",):
            system_instruction += "8. 你只能基于当前上传文档的内容回答。如果文档解析内容有限，请如实告知用户。\n"
        elif intent in ("research", "rag", "feed_research", "mixed"):
            system_instruction += "8. 如果证据不足，要明确说「当前相关材料中没有足够信息」。\n"
        approval_line = _approval_context_line(state)
        system_instruction += (
            f"{(9 if intent in ('research', 'rag', 'feed_research', 'mixed', 'document_qa') else 8)}. 如果只是问候或闲聊，要像正常助手一样回答；如果是记忆写入类请求，直接确认已保存即可。\n"
            f"{(10 if intent in ('research', 'rag', 'feed_research', 'mixed', 'document_qa') else 9)}. {approval_line}\n"
            f"{(11 if intent in ('research', 'rag', 'feed_research', 'mixed', 'document_qa') else 10)}. 输出简洁、结构化、可执行。中文为主。\n"
            "\n"
            "[Output Rules — 必须严格遵守]\n"
            "• 你必须输出用户可直接阅读的自然语言，可以使用 Markdown 标题、无序列表、加粗来组织内容。不要使用分割线 ---。\n"
            "• 严禁输出 JSON。严禁输出 Python dict。严禁输出 JavaScript object。\n"
            "• 严禁在你的回答中出现以下英文内部术语：evidence item / items、research question、information-gap opportunity、chunk、knowledge base、limited to available context、Based on X evidence items。\n"
            "• 严禁出现以下中文内部表达：基于当前知识库证据、证据显示、根据检索到的证据、当前知识库中没有找到足够证据、research question、information-gap。\n"
            "• 如果你需要引用文档内容，请直接用自然中文描述，不要把内部 payload 原样贴给用户。\n"
            "• 你的回答就是用户最终看到的全部内容，没有二次解析步骤。\n"
            "• 当你给出多个选项或「需要我帮你」列表时，请使用标准 Markdown 无序列表，每个选项一行。"
        )
        return system_instruction

    def _build_legacy_prompt(
        self,
        state: AgentRuntimeState,
        draft_answer: str,
        route_plan: dict[str, Any],
    ) -> str:
        """Fallback prompt when gssc_context is empty."""
        payload = {
            "user_input": state.get("user_input", ""),
            "intent": route_plan.get("intent", "chat"),
            "risk_level": route_plan.get("risk_level", "L0"),
            "needs_approval": route_plan.get("needs_approval", False),
            "draft_answer": draft_answer,
            "context_summary": (state.get("context") or {}).get("conversation_summary", ""),
            "feed_card": (state.get("context") or {}).get("feed_card", {}),
            "research": state.get("research_result") or state.get("research") or {},
            "rag": state.get("rag_result") or state.get("rag") or {},
            "artifacts": state.get("artifacts", []),
            "tool_result": state.get("tool_result") or state.get("tool_call") or {},
            "memory_updates_count": len(state.get("memory_updates", [])),
            "skill_drafts_count": len(state.get("skill_drafts", [])),
            "errors": state.get("errors", []),
        }
        # Build a plain-text summary of the runtime data (NOT JSON) so the
        # LLM does not mimic JSON in its response.
        context_summary = str(payload.get("context_summary", "") or "")
        feed_title = str((payload.get("feed_card") or {}).get("title", ""))
        research_summary = str((payload.get("research") or {}).get("summary", ""))
        rag_answer = str((payload.get("rag") or {}).get("answer", ""))
        artifact_titles = [str(a.get("title", "")) for a in (payload.get("artifacts") or [])[:3] if a.get("title")]
        tool_status = str((payload.get("tool_result") or {}).get("status", ""))
        runtime_context = (
            f"意图: {payload.get('intent', 'chat')} | 风险: {payload.get('risk_level', 'L0')}\n"
            + (f"会话摘要: {context_summary}\n" if context_summary else "")
            + (f"关联信息流: {feed_title}\n" if feed_title else "")
            + (f"研究摘要: {research_summary[:300]}\n" if research_summary else "")
            + (f"RAG 回答: {rag_answer[:300]}\n" if rag_answer else "")
            + (f"工具状态: {tool_status}\n" if tool_status else "")
            + (f"已有成果物: {', '.join(artifact_titles)}\n" if artifact_titles else "")
            + (f"错误: {len(payload.get('errors', []))} 条\n" if payload.get("errors") else "")
        )
        return (
            "你是信息差 Agent OS 的最终回复节点。请基于下面的运行上下文直接用自然语言回答用户。\n\n"
            f"运行上下文：\n{runtime_context}\n"
            f"用户输入：{payload.get('user_input', '')}\n\n"
            "你可以用自然中文说明你正在或已经做了什么，但只能给用户可读的简短执行摘要，不能泄露私密推理。\n"
            "如果只是问候或闲聊，要像正常助手一样回答，不要说 Agent Run 完成。\n"
            "如果没有真实执行研究、生成 Artifact 或外部工具动作，必须诚实说明，不要假装已经完成。\n"
            f"{_approval_context_line(state)}\n"
            "回答要贴合用户原话，优先给结论，然后给下一步可做什么。中文为主。\n"
            "\n"
            "[Output Rules — 必须严格遵守]\n"
            "• 你必须输出用户可直接阅读的自然语言，可以使用 Markdown 标题、列表、加粗来组织内容。不要使用分割线 ---。\n"
            "• 严禁输出 JSON。严禁输出 Python dict。严禁输出 JavaScript object。\n"
            "• 严禁在你的回答中出现 status、final_output、artifacts、memory_updates、skill_drafts、evidence 这些内部字段名。\n"
            "• 你的回答就是用户最终看到的全部内容，没有二次解析步骤。"
        )

    def _looks_like_internal_json(self, text: str) -> bool:
        """Detect whether the LLM output is an internal JSON payload."""
        stripped = text.lstrip()
        if not stripped.startswith("{"):
            return False
        internal_keys = [
            '"status"', '"final_output"', '"artifacts"',
            '"memory_updates"', '"skill_drafts"', '"evidence"',
            '"memory_writes"', '"agent_outputs"',
        ]
        head = stripped[:600]
        return any(k in head for k in internal_keys)

    def _extract_text_from_json_output(self, text: str) -> str:
        """Extract user-visible text from an LLM that output internal JSON."""
        import json as _json
        try:
            data = _json.loads(text)
            if not isinstance(data, dict):
                return ""
        except (_json.JSONDecodeError, TypeError):
            return ""
        for key in ("final_output", "answer", "content", "message", "text", "summary"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val
        # Last resort: if there's a markdown_report, use it
        report = data.get("markdown_report")
        if isinstance(report, str) and report.strip():
            return report
        return ""

    def _is_generic_draft_answer(self, value: str) -> bool:
        stripped = value.strip()
        normalized = stripped.rstrip(".\u3002").lower()
        if normalized in {"", "agent completed", "agent run completed", "agent runtime completed"}:
            return True
        # If the draft answer looks like JSON, treat it as generic so the LLM
        # generates a proper Markdown response instead of echoing the JSON.
        if self._looks_like_internal_json(stripped):
            return True
        return False

    def _fallback_final_answer(self, state: AgentRuntimeState, draft_answer: str) -> str:
        user_input = str(state.get("user_input") or "").strip()
        route_plan = state.get("route_plan") or {}
        intent = route_plan.get("intent", "chat")
        # ── Memory write: confirm the save ─────────────────────────
        mem_result = state.get("memory_write_result") or {}
        if mem_result.get("success"):
            content = mem_result.get("content", "")
            return f"已记住：{content}"
        if intent == "memory" or self._is_explicit_memory_write(user_input):
            return f"已记住：{user_input}"
        if any(token in user_input for token in ("你好", "您好", "你是谁", "你是誰")) or user_input.lower() in {"hi", "hello", "hey"}:
            return "你好，我是你的信息差 Agent OS 助手。你可以让我分析首页信息差、做深度研究、生成报告或代码成果，也可以把反复使用的流程沉淀成长期记忆和 Skill。"
        normalized = draft_answer.strip().lower().rstrip(".。")
        if draft_answer and normalized not in {"agent run completed", "agent runtime completed"}:
            return draft_answer.strip()
        if intent == "research":
            return "我已识别这是一个研究任务，并完成了需求判断和执行规划。当前没有产生可验证的完整研究结果，因此不会假装已经完成深度研究。你可以继续指定研究范围，我会进入资料检索和结构化报告生成。"
        if intent == "artifact":
            return "我已识别这是一个成果生成任务，并完成了初步规划。当前还没有生成实际 Artifact。你可以继续指定要生成文档、报告、网站还是代码。"
        return "我已经完成本次请求的基础判断和上下文检查。你可以继续补充目标，我会沿用当前会话上下文继续处理。"

    def _message_content(self, message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        return str(content)
    def _rule_home_intent(self, user_input: str, feed_card_id: Any) -> dict[str, Any]:
        route_plan = plan_route(
            user_input=user_input,
            feed_card_id=feed_card_id,
            forced_route=self.payload.get("route"),
            forced_intent=self.payload.get("intent"),
        )
        result = HomeIntentResult(
            intent=route_plan.get("intent", "chat"),
            confidence=0.72 if route_plan.get("reason") != "default_chat_route" else 0.5,
            risk_level=route_plan.get("risk_level", "L0"),
            needs_approval=route_plan.get("needs_approval", False),
            needs_clarification=False,
            required_agents=route_plan.get("route", []),
            expected_output=route_plan.get("expected_output", "answer"),
            reason_summary=route_plan.get("reason", "default_chat_route"),
            suggested_route_hints=route_plan.get("route", []),
            fallback_used=False,
            raw_intent_source="rule",
        )
        return result.to_home_intent_dict()

    def _apply_rule_risk_floor(self, llm_intent: HomeIntentResult, rule_intent: dict[str, Any]) -> dict[str, Any]:
        order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
        data = llm_intent.to_home_intent_dict()
        rule_risk = str(rule_intent.get("risk_level", "L0"))
        llm_risk = str(data.get("risk_level", "L0"))
        if order.get(rule_risk, 0) > order.get(llm_risk, 0):
            data["risk_level"] = rule_risk
            data["needs_approval"] = rule_risk in {"L3", "L4"} or bool(data.get("needs_approval"))
            data["reason_summary"] = f"{data.get('reason_summary', '')}；规则风险兜底提升为 {rule_risk}".strip("；")
            data["reasoning_summary"] = data["reason_summary"]
        return data

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

    def _is_explicit_memory_write(self, user_input: str) -> bool:
        """Check whether the user is explicitly asking to remember something."""
        if not user_input:
            return False
        text = user_input.strip().lower()
        return any(pattern in text for pattern in self._MEMORY_WRITE_PATTERNS)

    def _extract_memory_from_user_input(self, text: str) -> str:
        """Extract a clean memory sentence from a user input containing '记住'."""
        import re
        content = text.strip()
        # Remove trailing "记住" / "帮我记" etc.
        content = re.sub(r"[，,]*\s*(记住|帮我记|记一下|记下来|别忘了|下次记住|保存下来)[。.]*$", "", content)
        content = re.sub(r"^[记住：:记住:\s]+", "", content)
        content = content.strip()
        if not content:
            return text.strip()
        # Normalize to third-person or factual statement
        # Convert first-person "我" to "用户" for persistent memory
        if content.startswith("我"):
            content = "用户" + content[1:]
        return content

    def _draft_title(self, user_input: str) -> str:
        title = " ".join(str(user_input).strip().split())[:40]
        return f"Reusable Agent workflow: {title}" if title else "Reusable Agent workflow"


# ── Module-level helpers ─────────────────────────────────────────

_SENSITIVE_ARG_KEYS = {"password", "token", "secret", "api_key", "access_key", "smtp_password", "credential", "auth"}


def _approval_context_line(state: dict[str, Any]) -> str:
    """Return the approval-instruction line for the final LLM prompt.

    On resume (after approve/reject), the LLM must NOT say "需要审批".
    Instead it should describe what actually happened.
    """
    resume_context = str(state.get("_resume_context") or state.get("resume_token") or "")
    tool_name = state.get("pending_tool_name") or (state.get("tool_call") or {}).get("tool_name") or ""
    tool_status = (state.get("tool_call") or {}).get("status", "")
    tool_error = state.get("_tool_error") or (state.get("tool_result") or {}).get("error", "")

    if resume_context.startswith("rejected:") or tool_status == "rejected":
        return "用户已取消该操作，没有执行。请用自然语言说明已取消，并给出替代建议（如有）。不要声称已经执行。"
    if resume_context.startswith("approved:") and tool_status == "completed":
        return "该操作已获得用户批准并已执行成功。请基于工具执行结果回答，不要声称需要审批。"
    if resume_context.startswith("failed:") or tool_error:
        disp_name = _tool_display_name_local(tool_name)
        return f"该操作已获得用户批准但执行失败（{disp_name}: {tool_error[:200]}）。请用自然语言说明失败原因，不要声称已经执行成功，也不要声称需要审批。"
    if state.get("status") == "resuming":
        return "该操作已获得批准。请基于工具结果回答，不要声称需要审批。"
    # Normal path (no resume context)
    return "如果涉及 L3/L4 风险动作，说明需要审批，不能声称已经执行。"


def _tool_display_name_local(tool_name: str) -> str:
    names = {
        "email.send": "发送邮件",
        "local_file.write": "写入文件",
        "local_file.append": "追加文件",
        "local_file.delete": "删除文件",
        "local_file.read": "读取文件",
        "local_file.list": "列出文件",
    }
    return names.get(tool_name, tool_name)


def _sanitize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of tool_args with sensitive fields redacted."""
    if not args:
        return {}
    safe: dict[str, Any] = {}
    for key, value in args.items():
        lower_key = key.lower()
        if any(s in lower_key for s in _SENSITIVE_ARG_KEYS):
            safe[key] = "[redacted]"
        elif isinstance(value, str) and len(value) > 500:
            safe[key] = value[:500] + "..."
        else:
            safe[key] = value
    return safe
