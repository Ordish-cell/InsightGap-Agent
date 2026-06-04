from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_step
from src.web_app.agent.runtime.planner import plan_route
from src.web_app.agent.runtime.router import route_user_input
from src.web_app.agent.runtime.state import AgentRuntimeState, append_error, append_output, mark_completed
from src.web_app.mcp.tool_router import infer_tool
from src.web_app.context.builder import ContextBuilder
from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
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
        return state

    async def home_intent_react(self, state: AgentRuntimeState) -> AgentRuntimeState:
        user_input = state.get("user_input", "") or state.get("query", "")
        page_context = self.payload.get("page_context") or state.get("page_context") or {}
        feed_card_id = self.payload.get("feed_card_id") or page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
        route_plan = plan_route(
            user_input=user_input,
            feed_card_id=feed_card_id,
            forced_route=self.payload.get("route"),
            forced_intent=self.payload.get("intent"),
        )
        home_intent = {
            "detected_intent": route_plan.get("intent", "chat"),
            "confidence": 0.72 if route_plan.get("reason") != "default_chat_route" else 0.5,
            "risk_level": route_plan.get("risk_level", "L0"),
            "required_capabilities": [node.replace("_agent", "") for node in route_plan.get("route", []) if node.endswith("_agent")],
            "expected_output": route_plan.get("expected_output", "answer"),
            "needs_approval": route_plan.get("needs_approval", False),
            "needs_clarification": False,
            "reasoning_summary": route_plan.get("reason", "default_chat_route"),
            "suggested_route_hints": route_plan.get("route", []),
        }
        state["home_intent"] = home_intent
        record_step(self.db, state["run_id"], "home_intent_react", "triage_intent", {"user_input": user_input, "page_context": page_context}, {"home_intent": home_intent})
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
        profile = ProfileRepository(self.db).get_or_create_default(state["user_id"])
        memories = memory_service.search_memory(state["user_id"], state.get("user_input", ""), min_importance=0.2, db=self.db)[:5]
        page_context = self.payload.get("page_context") or state.get("page_context") or {}
        feed_card_id = self.payload.get("feed_card_id") or page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
        feed_card_context = self._load_feed_card_context(state["user_id"], feed_card_id)

        # Generate conversation summary from recent agent steps
        conversation_summary = self._build_conversation_summary(state["run_id"])

        # Generate checkpoint summary from current state
        checkpoint_summary = self._build_checkpoint_summary(state)

        # Get dynamic preferences from user growth engine
        dynamic_prefs = user_growth_service.build_dynamic_preference_profile(
            state["user_id"], self.db, route=route,
        )

        builder = ContextBuilder(route=route)
        context_text, gssc_debug = builder.build_with_debug({
            "task": state.get("user_input", ""),
            "route": route,
            "profile": {"segment": profile.segment, "goals": profile.goals, "interests": profile.explicit_interests},
            "memory": memories,
            "feed_card": feed_card_context,
            "page_context": page_context,
            "conversation_summary": conversation_summary,
            "checkpoint_summary": checkpoint_summary,
            "dynamic_preferences": dynamic_prefs.get("preference_summary", ""),
            "output_contract": "Return structured status, final_output, artifacts, memory_updates, skill_drafts, and evidence when available.",
        })
        state["context"] = {
            "gssc_context": context_text,
            "memory_count": len(memories),
            "feed_card": feed_card_context,
            "page_context": page_context,
            "gssc_debug": gssc_debug,
            "conversation_summary": conversation_summary,
            "checkpoint_summary": checkpoint_summary,
        }
        record_step(self.db, state["run_id"], "context_builder", "context",
                    {"route": route, "feed_card_id": feed_card_id},
                    {"memory_count": len(memories), "feed_card_loaded": bool(feed_card_context),
                     "gssc_debug": gssc_debug, "context": context_text})
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
        status = state.get("status") or ("failed" if state.get("error") else "completed")
        if state.get("route") == "approval":
            status = "waiting_approval"
        state["status"] = status
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

        route_plan = plan_route(
            user_input=user_input,
            feed_card_id=feed_card_id,
            forced_route=self.payload.get("route"),
            forced_intent=self.payload.get("intent"),
        )
        state["route_plan"] = route_plan
        state["route"] = route_plan.get("intent", "chat")  # legacy compat
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
        record_step(self.db, state["run_id"], "planner", "plan_route",
                    {"user_input": user_input, "feed_card_id": feed_card_id},
                    {"route_plan": route_plan})
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
        except Exception as exc:
            append_error(state, "research_agent", str(exc))
            record_step(self.db, state["run_id"], "research_agent", "deep_research",
                        {}, {"error": str(exc)}, status="failed")
        mark_completed(state, "research_agent")
        return state

    async def rag_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """RAG Agent: retrieve from user's knowledge base."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "rag_agent")
            return state
        try:
            result = rag_service.ask(state["user_id"], state.get("user_input", ""),
                                     top_k=int(self.payload.get("top_k", 5)))
            state["rag"] = result
            state["rag_result"] = result
            append_output(state, "rag_agent", {"answer": result.get("answer", ""),
                          "evidence_count": len(result.get("evidence", []))})
            record_step(self.db, state["run_id"], "rag_agent", "rag_ask",
                        {"query": state.get("user_input", "")},
                        {"answer_mode": result.get("answer_mode"),
                         "evidence_count": len(result.get("evidence", []))})
        except Exception as exc:
            append_error(state, "rag_agent", str(exc))
            state["rag_result"] = {"answer": "", "evidence": [], "error": str(exc)}
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
        except Exception as exc:
            append_error(state, "artifact_agent", str(exc))
        mark_completed(state, "artifact_agent")
        return state

    async def tool_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """MCP Tool Agent: infer and execute tools with approval guard."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "tool_agent")
            return state
        try:
            route_plan = state.get("route_plan") or {}
            if route_plan.get("needs_approval"):
                state["status"] = "waiting_approval"
                state["tool_result"] = {
                    "status": "approval_required",
                    "tool_name": "inferred_from_input",
                    "risk_level": route_plan.get("risk_level", "L3"),
                    "summary": route_plan.get("reason", ""),
                    "arguments_preview": {"user_input": state.get("user_input", "")},
                }
                append_output(state, "tool_agent", state["tool_result"])
                record_step(self.db, state["run_id"], "tool_agent", "mcp_approval_required",
                            {"route_plan": route_plan}, {"tool_result": state["tool_result"]})
                mark_completed(state, "tool_agent")
                return state

            tool_name, tool_input = infer_tool(state.get("user_input", ""), self.payload)
            if not tool_name:
                append_error(state, "tool_agent", "tool_not_found")
            else:
                result = mcp_service.call_tool(self.db, state["user_id"], tool_name, tool_input,
                                               agent_run_id=state["run_id"],
                                               dry_run=bool(self.payload.get("dry_run", False)))
                state["tool_call"] = result
                state["tool_result"] = result
                append_output(state, "tool_agent", {"tool_name": tool_name, "status": result.get("status")})
                record_step(self.db, state["run_id"], "tool_agent", "mcp_call",
                            {"tool_name": tool_name}, {"status": result.get("status")})
        except Exception as exc:
            append_error(state, "tool_agent", str(exc))
        mark_completed(state, "tool_agent")
        return state

    async def memory_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Memory Agent: conditionally write memories based on task value."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "memory_agent")
            return state
        try:
            route_plan = state.get("route_plan") or {}
            intent = route_plan.get("intent", "chat")
            # Only write memory for high-value tasks, not casual chat
            if intent in ("chat",):
                mark_completed(state, "memory_agent")
                return state

            user_input = state.get("user_input", "")
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
        except Exception as exc:
            append_error(state, "memory_agent", str(exc))
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
        except Exception as exc:
            append_error(state, "skill_agent", str(exc))
        mark_completed(state, "skill_agent")
        return state

    async def final_response(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Final Response: aggregate all agent outputs into final_answer and final_payload."""
        route_plan = state.get("route_plan") or {}
        intent = route_plan.get("intent", "chat")

        # Build final_answer from all agent outputs
        answer_parts = []
        research = state.get("research_result") or state.get("research") or {}
        if research.get("summary"):
            answer_parts.append(research["summary"])
        rag = state.get("rag_result") or state.get("rag") or {}
        if rag.get("answer"):
            answer_parts.append(rag["answer"])
        if state.get("final_output") and not answer_parts:
            answer_parts.append(state["final_output"])

        if answer_parts:
            final_answer = "\n\n".join(answer_parts)
        elif state.get("status") == "waiting_approval":
            final_answer = state.get("final_output") or "Approval required before continuing."
        else:
            final_answer = state.get("final_output") or "Agent run completed."

        state["final_answer"] = final_answer
        state["final_output"] = final_answer  # legacy compat

        # Build structured final_payload for frontend
        errors = state.get("errors", [])
        status = state.get("status") or ("failed" if errors else "completed")

        final_payload = {
            "run_id": str(state.get("run_id", "")),
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
        state["final_payload"] = final_payload
        state["status"] = status

        record_step(self.db, state["run_id"], "final_response", "aggregate",
                    {"intent": intent},
                    {"status": status, "answer_len": len(final_answer),
                     "artifact_count": len(state.get("artifacts", [])),
                     "error_count": len(errors)})
        return state

    # ── Helper methods ──────────────────────────────────────────────

    def _build_conversation_summary(self, run_id: int) -> str:
        """Build a short conversation summary from recent agent steps of this run."""
        try:
            from src.web_app.db.repositories.agent_repository import AgentRunRepository
            repo = AgentRunRepository(self.db)
            steps = repo.list_steps(run_id)[-6:]  # last 6 steps
        except Exception:
            steps = []
        if not steps:
            return ""
        step_texts = []
        for step in steps:
            node = getattr(step, "node_name", "") or ""
            status = getattr(step, "status", "") or ""
            output = getattr(step, "output", {}) or {}
            if isinstance(output, dict):
                route = output.get("route", "")
                if route:
                    step_texts.append(f"{node}(→{route}/{status})")
                else:
                    step_texts.append(f"{node}({status})")
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

    def _draft_title(self, user_input: str) -> str:
        title = " ".join(str(user_input).strip().split())[:40]
        return f"Reusable Agent workflow: {title}" if title else "Reusable Agent workflow"
