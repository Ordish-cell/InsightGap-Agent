from __future__ import annotations

import json

from src.web_app.agent.runtime.events import queue_stream_event
from src.web_app.agent.runtime.node_groups.base import *
from src.web_app.agent.runtime.state_delta import record_agent_node_result


def _tool_node_result_updates(state: AgentRuntimeState) -> dict[str, Any]:
    keys = (
        "tool_call",
        "tool_result",
        "approval_payload",
        "approval_required",
        "pending_approval_id",
        "pending_tool_name",
        "pending_tool_args",
        "pending_tool_call_id",
        "resume_token",
        "final_output",
    )
    return {key: state.get(key) for key in keys if key in state}


def _tool_event_id(state: AgentRuntimeState, tool_name: str) -> str:
    count = len(state.get("tool_calls") or [])
    return f"run-{state.get('run_id', 'unknown')}-tool-{count + 1}-{tool_name}"


def _tool_output_preview(result: dict[str, Any] | None, max_chars: int = 700) -> str:
    if not result:
        return ""
    output = result.get("output") if isinstance(result, dict) else None
    value: Any = output if output not in (None, {}) else result
    if isinstance(value, dict):
        value = {
            key: item
            for key, item in value.items()
            if not str(key).startswith("_") and not any(secret in str(key).lower() for secret in _SENSITIVE_ARG_KEYS)
        }
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    text = " ".join(text.split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _queue_tool_event(
    state: AgentRuntimeState,
    event_type: str,
    *,
    tool_call_id: str,
    tool_name: str,
    args_preview: dict[str, Any] | None = None,
    output_preview: str = "",
    status: str = "",
    error: str = "",
    tool_call_record_id: int | str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "toolCallId": tool_call_id,
        "tool_name": tool_name,
        "toolName": tool_name,
        "status": status,
    }
    if args_preview is not None:
        payload["args_preview"] = args_preview
        payload["argsPreview"] = args_preview
    if output_preview:
        payload["output_preview"] = output_preview
        payload["outputPreview"] = output_preview
    if error:
        payload["error"] = error[:700]
    if tool_call_record_id is not None:
        payload["tool_call_record_id"] = tool_call_record_id
    queue_stream_event(
        state.get("_stream_queue"),
        event_type,
        payload,
        run_id=state.get("run_id"),
        thread_id=state.get("thread_id", ""),
        node_name="tool_agent",
    )


class AgentNodesMixin:
    async def research_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Research Agent: execute deep research via ResearchService."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "research_agent")
            record_agent_node_result(
                state,
                node="research_agent",
                updates={},
                status="skipped",
                summary="Skipped because route is approval or blocked.",
            )
            return state
        try:
            route_plan = state.get("route_plan") or {}
            force_engine = self.payload.get("force_engine")
            if not route_plan.get("explicit_research"):
                force_engine = "fallback"
            request = ResearchRequest(
                query=self.payload.get("query") or state.get("user_input", ""),
                depth=self.payload.get("depth", "standard"),
                save_artifact=self.payload.get("save_artifact", True),
                write_memory=self.payload.get("write_memory", True),
                create_skill_draft=self.payload.get("create_skill_draft", True),
                force_engine=force_engine,
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
            evidence = list(result.get("evidence", []) or [])
            engine = ((result.get("metadata") or {}).get("engine")
                      or (result.get("metadata_json") or {}).get("engine")
                      or ("open_deep_research" if route_plan.get("explicit_research") else "fallback_researcher"))
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "research_agent"),
                agent="research_agent",
                status="ok" if result.get("status") != "failed" else "failed",
                confidence=0.85 if evidence else 0.65,
                summary=result.get("summary", ""),
                findings=[str(item) for item in (result.get("findings", []) or [])],
                evidence=evidence,
                artifacts=[{"id": result["artifact_id"], "type": "research_report"}] if result.get("artifact_id") else [],
                warnings=[] if route_plan.get("explicit_research") else [f"research_engine={engine}"],
            ))
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "research_agent"),
                agent="research_agent",
                status="failed",
                confidence=0.0,
                summary="Research agent failed.",
                errors=[str(exc)],
                warnings=["research_failed"],
            ))
            record_step(self.db, state["run_id"], "research_agent", "deep_research",
                        {}, {"error": str(exc)}, status="failed")
        emit_visible_thought(self.db, state, "research_agent")
        mark_completed(state, "research_agent")
        record_agent_node_result(
            state,
            node="research_agent",
            updates={
                "research": state.get("research"),
                "research_result": state.get("research_result"),
                "artifacts": state.get("artifacts", []),
                "skill_drafts": state.get("skill_drafts", []),
            },
            summary=(state.get("research_result") or {}).get("summary", ""),
        )
        return state

    async def rag_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """RAG Agent: retrieve from user's knowledge base."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "rag_agent")
            record_agent_node_result(
                state,
                node="rag_agent",
                updates={},
                status="skipped",
                summary="Skipped because route is approval or blocked.",
            )
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
            prepared_rag = ((state.get("parallel_read_results") or {}).get("rag_prepare") or {})
            prepared_evidence = list(prepared_rag.get("evidence") or []) if prepared_rag.get("status") == "ok" else []
            prepare_no_evidence_used = False
            can_reuse_empty_prepare = (
                prepared_rag.get("status") == "ok"
                and bool(prepared_rag.get("search_attempted"))
                and not prepared_evidence
                and intent not in ("document_qa",)
                and not overview
                and not doc_ids_raw
            )
            if prepared_evidence:
                from src.web_app.context.builder import ContextBuilder as _RagContextBuilder
                context = _RagContextBuilder().build({
                    "task": user_input_str,
                    "evidence": prepared_evidence,
                    "output_contract": "Answer only from evidence and cite chunk_id/source_title.",
                })
                result = {
                    "answer": rag_service._extractive_answer(user_input_str, prepared_evidence),
                    "answer_mode": "extractive_fallback",
                    "evidence": prepared_evidence,
                    "context": {
                        "gssc_used": True,
                        "selected_chunks": len(prepared_evidence),
                        "token_estimate": max(1, len(context) // 4),
                        "embedding_model": resolve_model_name("embedding").model,
                        "answer_model": resolve_model_name("rag").model,
                        "prepared_evidence_used": True,
                    },
                    "_parallel_read_evidence_used": True,
                }
            elif can_reuse_empty_prepare:
                prepare_no_evidence_used = True
                result = {
                    "answer": "",
                    "answer_mode": "no_evidence_from_prepare",
                    "evidence": [],
                    "needs_general_fallback": True,
                    "context": {
                        "gssc_used": False,
                        "selected_chunks": 0,
                        "embedding_model": resolve_model_name("embedding").model,
                        "answer_model": resolve_model_name("rag").model,
                        "prepared_evidence_used": True,
                        "prepared_no_evidence_used": True,
                    },
                    "_parallel_read_no_evidence_used": True,
                }
            elif intent in ("document_qa",) and overview and doc_ids_raw:
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
            if result.get("needs_general_fallback"):
                from src.web_app.services.rag_service import _answer_from_general_llm
                try:
                    fallback_answer = await _answer_from_general_llm(user_input_str)
                    result["answer"] = fallback_answer
                    result["answer_mode"] = "general_knowledge_fallback"
                    result["_fallback_used"] = True
                except Exception: pass
            prefetched_evidence = (((state.get("prefetch_results") or {}).get("rag") or {}).get("evidence") or [])
            if prefetched_evidence and not result.get("evidence"):
                result["evidence"] = list(prefetched_evidence)
                result["_prefetch_evidence_used"] = True
            state["rag"] = result
            state["rag_result"] = result
            append_output(state, "rag_agent", {"answer": result.get("answer", ""),
                          "evidence_count": len(result.get("evidence", []))})
            evidence = list(result.get("evidence", []) or [])
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "rag_agent"),
                agent="rag_agent",
                status="ok",
                confidence=0.85 if evidence else 0.45,
                summary=result.get("answer", ""),
                findings=[result.get("answer", "")] if result.get("answer") else [],
                evidence=evidence,
                warnings=[] if evidence else ["evidence_missing"],
            ))
            record_step(self.db, state["run_id"], "rag_agent", "rag_ask",
                        {"query": user_input_str, "intent": intent, "overview": overview},
                        {"answer_mode": result.get("answer_mode"),
                         "evidence_count": len(result.get("evidence", [])),
                         "prepare_no_evidence_used": prepare_no_evidence_used})
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "rag_agent"),
                agent="rag_agent",
                status="failed",
                confidence=0.0,
                summary="",
                errors=[str(exc)],
                warnings=["rag_failed"],
            ))
        emit_visible_thought(self.db, state, "rag_agent")
        mark_completed(state, "rag_agent")
        record_agent_node_result(
            state,
            node="rag_agent",
            updates={
                "rag": state.get("rag"),
                "rag_result": state.get("rag_result"),
            },
            summary=(state.get("rag_result") or {}).get("answer", ""),
        )
        return state

    async def artifact_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Artifact Agent: generate and save a document artifact."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "artifact_agent")
            record_agent_node_result(
                state,
                node="artifact_agent",
                updates={},
                status="skipped",
                summary="Skipped because route is approval or blocked.",
            )
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "artifact_agent"),
                agent="artifact_agent",
                status="ok",
                confidence=0.9,
                summary=f"Created artifact {item.title}.",
                artifacts=[artifact],
            ))
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
            state["artifact_result"] = {"error": str(exc)}
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "artifact_agent"),
                agent="artifact_agent",
                status="failed",
                confidence=0.0,
                summary="Artifact generation failed.",
                errors=[str(exc)],
                warnings=["artifact_failed"],
            ))
        emit_visible_thought(self.db, state, "artifact_agent")
        mark_completed(state, "artifact_agent")
        record_agent_node_result(
            state,
            node="artifact_agent",
            updates={
                "artifact_result": state.get("artifact_result"),
                "artifacts": state.get("artifacts", []),
            },
            summary=(state.get("artifact_result") or {}).get("title", ""),
        )
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
            record_agent_node_result(
                state,
                node="tool_agent",
                updates=_tool_node_result_updates(state),
                status="skipped",
                summary="Skipped because route is approval or blocked.",
            )
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "tool_agent"),
                agent="tool_agent",
                status="ok",
                confidence=0.9,
                summary="Tool action completed after approval.",
                tool_calls=[state.get("tool_call", {})] if state.get("tool_call") else [],
            ))
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            record_agent_node_result(
                state,
                node="tool_agent",
                updates=_tool_node_result_updates(state),
                summary="Tool action completed after approval.",
            )
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "tool_agent"),
                agent="tool_agent",
                status="failed",
                confidence=0.0,
                summary="Tool action failed after approval.",
                tool_calls=[state.get("tool_call", {})] if state.get("tool_call") else [],
                errors=[str((state.get("tool_result") or {}).get("message") or state.get("_tool_error") or "tool_failed")],
                warnings=["tool_failed"],
            ))
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            record_agent_node_result(
                state,
                node="tool_agent",
                updates=_tool_node_result_updates(state),
                summary="Tool action failed after approval.",
            )
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "tool_agent"),
                agent="tool_agent",
                status="denied",
                confidence=0.8,
                summary="Tool action was rejected by the user.",
                tool_calls=[state.get("tool_call", {})] if state.get("tool_call") else [],
                warnings=["tool_rejected"],
            ))
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            record_agent_node_result(
                state,
                node="tool_agent",
                updates=_tool_node_result_updates(state),
                summary="Tool action was rejected by the user.",
            )
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "tool_agent"),
                agent="tool_agent",
                status="ok",
                confidence=0.7,
                summary="Tool resume state was accepted.",
                tool_calls=[state.get("tool_call", {})] if state.get("tool_call") else [],
            ))
            emit_visible_thought(self.db, state, "tool_agent")
            mark_completed(state, "tool_agent")
            record_agent_node_result(
                state,
                node="tool_agent",
                updates=_tool_node_result_updates(state),
                summary="Tool resume state was accepted.",
            )
            return state

        try:
            route_plan = state.get("route_plan") or {}
            user_text = state.get("user_input", "")

            # ── LLM tool selection (first entry only, reused on resume) ──
            llm_selection = None
            if state.get("llm_tool_selection"):
                # Reuse stored result (dict form from model_dump)
                try:
                    llm_selection = LLMToolSelectionResult.model_validate(state["llm_tool_selection"])
                except Exception:
                    llm_selection = None
            elif state.get("status") != "resuming":
                # First entry: call LLM
                try:
                    available_tools = [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "risk_level": spec.safety_level,
                            "requires_approval": spec.requires_approval,
                            "aliases": spec.aliases,
                            "input_schema": spec.input_schema,
                        }
                        for spec in BUILTIN_TOOLS
                        if spec.enabled
                    ]
                    llm_selection = llm_select_tools(
                        self.db,
                        run_id=state["run_id"],
                        thread_id=state.get("thread_id", ""),
                        user_id=state["user_id"],
                        user_input=user_text,
                        available_tools=available_tools,
                    )
                    state["llm_tool_selection"] = llm_selection.model_dump()
                    logger.info(
                        "[LLM_TOOL_SELECT_DEBUG] available_tools_count=%d user_text=%.200s "
                        "confidence=%.2f tool_name=%s route=%s missing_fields=%s",
                        len(available_tools), user_text[:200],
                        llm_selection.confidence,
                        llm_selection.tool_calls[0].name if llm_selection.tool_calls else "",
                        llm_selection.route,
                        str(llm_selection.missing_fields)[:200],
                    )
                except Exception as exc:
                    logger.warning("[LLM_TOOL_SELECT_DEBUG] llm_select_tools failed, falling back to keywords: %s", exc)
                    state["llm_tool_selection"] = None
                    llm_selection = None

            tool_name, tool_input = infer_tool(
                user_text,
                {**self.payload, "intent": route_plan.get("intent")},
                llm_result=llm_selection,
            )

            # ── Memory-guard: memory-like input must never trigger email.send ──
            if _is_memory_like_input(user_text) and tool_name == "email.send":
                logger.warning("[MEMORY_GUARD] Blocked false-positive email.send for memory-like input")
                tool_name = None
                tool_input = {}

            # ── Hard fallback guard: obvious email intent must never become tool_not_found ──
            if not tool_name and _is_obvious_email_intent(self.db, user_text, route_plan) and not _is_memory_like_input(user_text):
                logger.warning(
                    "[TOOL_NOT_FOUND_DEBUG] requested_tool=email.send normalized_tool=email.send "
                    "registered_tools=[email.send,...] user_text=%.200s — forcing email.send fallback",
                    user_text[:200],
                )
                tool_name = "email.send"
                tool_input = _build_email_input(user_text, self.payload)

            if not tool_name:
                logger.warning(
                    "[TOOL_NOT_FOUND_DEBUG] requested_tool=<none> normalized_tool=<none> "
                    "registered_tools=%s planner_raw_output=%s user_text=%.200s",
                    list(t.name for t in BUILTIN_TOOLS if t.enabled),
                    str(route_plan)[:200],
                    user_text[:200],
                )
                append_error(state, "tool_agent", "tool_not_found")
                append_agent_result(state, AgentResult(
                    task_id=task_id_for_agent(state, "tool_agent"),
                    agent="tool_agent",
                    status="failed",
                    confidence=0.0,
                    summary="No matching tool was found.",
                    errors=["tool_not_found"],
                    warnings=["tool_not_found"],
                ))
                mark_completed(state, "tool_agent")
                record_agent_node_result(
                    state,
                    node="tool_agent",
                    updates=_tool_node_result_updates(state),
                    summary="No matching tool was found.",
                )
                return state

            # ── Missing fields guard: stop BEFORE approval/execution ──
            cleaned_args, missing = validate_tool_input(tool_name, tool_input)
            if missing:
                answer = _build_missing_fields_answer(tool_name, cleaned_args, missing)
                state["final_output"] = answer
                state["tool_call"] = {
                    "status": "missing_fields",
                    "tool_name": tool_name,
                    "provided_args": cleaned_args,
                    "missing_fields": missing,
                }
                append_agent_result(state, AgentResult(
                    task_id=task_id_for_agent(state, "tool_agent"),
                    agent="tool_agent",
                    status="skipped",
                    confidence=0.7,
                    summary=f"Tool {tool_name} is missing required fields.",
                    tool_calls=[state["tool_call"]],
                    warnings=["tool_missing_fields"],
                ))
                logger.info(
                    "[LLM_TOOL_SELECT_DEBUG] tool=%s missing_fields=%s — stopping for user clarification",
                    tool_name, str(missing)[:200],
                )
                emit_visible_thought(self.db, state, "tool_agent")
                mark_completed(state, "tool_agent")
                record_agent_node_result(
                    state,
                    node="tool_agent",
                    updates=_tool_node_result_updates(state),
                    summary=f"Tool {tool_name} is missing required fields.",
                )
                return state

            client_tool_call_id = _tool_event_id(state, tool_name)
            safe_tool_args = _sanitize_tool_args(tool_name, tool_input)
            _queue_tool_event(
                state,
                "tool_call_started",
                tool_call_id=client_tool_call_id,
                tool_name=tool_name,
                args_preview=safe_tool_args,
                status="running",
            )

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
                    "tool_args": safe_tool_args,
                    "preview": result.get("output", {}),
                    "run_id": state["run_id"],
                    "user_id": state["user_id"],
                    "title": f"需要你确认：{tool_name}",
                    "actions": ["approve", "reject"],
                }
                append_output(state, "tool_agent", {"tool_name": tool_name, "status": "waiting_approval"})
                append_agent_result(state, AgentResult(
                    task_id=task_id_for_agent(state, "tool_agent"),
                    agent="tool_agent",
                    status="needs_approval",
                    confidence=0.8,
                    summary=f"Tool {tool_name} requires approval.",
                    tool_calls=[result],
                    warnings=["approval_pending"],
                ))
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
                record_agent_node_result(
                    state,
                    node="tool_agent",
                    updates=_tool_node_result_updates(state),
                    summary=f"Tool {tool_name} requires approval.",
                )
                return state
            elif result["status"] in {"failed", "blocked"}:
                _queue_tool_event(
                    state,
                    "tool_call_failed",
                    tool_call_id=client_tool_call_id,
                    tool_name=tool_name,
                    output_preview=_tool_output_preview(result),
                    status=str(result.get("status") or "failed"),
                    error=str(result.get("error") or "tool_failed"),
                    tool_call_record_id=result.get("id"),
                )
                state["status"] = "failed"
                state["error"] = result.get("error", "")
                state["final_output"] = f"工具 {tool_name} 失败: {result.get('error', 'unknown')}"
                append_error(state, "tool_agent", result.get("error", "tool_failed"))
                append_agent_result(state, AgentResult(
                    task_id=task_id_for_agent(state, "tool_agent"),
                    agent="tool_agent",
                    status="denied" if result["status"] == "blocked" else "failed",
                    confidence=0.0,
                    summary=f"Tool {tool_name} was denied." if result["status"] == "blocked" else f"Tool {tool_name} failed.",
                    tool_calls=[result],
                    errors=[str(result.get("error") or "tool_failed")],
                    warnings=["tool_denied"] if result["status"] == "blocked" else ["tool_failed"],
                ))
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
                _queue_tool_event(
                    state,
                    "tool_call_completed",
                    tool_call_id=client_tool_call_id,
                    tool_name=tool_name,
                    output_preview=_tool_output_preview(result),
                    status=str(result.get("status") or "completed"),
                    tool_call_record_id=result.get("id"),
                )
                append_output(state, "tool_agent", {"tool_name": tool_name, "status": result.get("status")})
                append_agent_result(state, AgentResult(
                    task_id=task_id_for_agent(state, "tool_agent"),
                    agent="tool_agent",
                    status="ok",
                    confidence=0.9,
                    summary=f"Tool {tool_name} completed with status {result.get('status')}.",
                    tool_calls=[result],
                ))
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "tool_agent"),
                agent="tool_agent",
                status="failed",
                confidence=0.0,
                summary="Tool agent failed.",
                errors=[str(exc)],
                warnings=["tool_agent_failed"],
            ))
        emit_visible_thought(self.db, state, "tool_agent")
        mark_completed(state, "tool_agent")
        record_agent_node_result(
            state,
            node="tool_agent",
            updates=_tool_node_result_updates(state),
            summary=(state.get("tool_call") or {}).get("tool_name", "") if isinstance(state.get("tool_call"), dict) else "",
        )
        return state

    async def memory_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Memory Agent: write semantic memories for explicit user requests,
        and conditionally extract memories for high-value tasks."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "memory_agent")
            record_agent_node_result(
                state,
                node="memory_agent",
                updates={},
                status="skipped",
                summary="Skipped because route is approval or blocked.",
            )
            return state
        try:
            route_plan = state.get("route_plan") or {}
            intent = route_plan.get("intent", "chat")
            user_input = state.get("user_input", "")
            thresholds = {
                "semantic_importance": 0.88,
                "semantic_confidence": 0.85,
                "episodic_importance": 0.85,
                "episodic_confidence": 0.80,
            }

            # ── Explicit memory write: user says "记住" / "帮我记" etc. ──
            if self._is_explicit_memory_write(user_input):
                memory_content = self._extract_memory_from_user_input(user_input)
                # Store candidates BEFORE writing
                state["memory_candidates"] = [{
                    "content": memory_content,
                    "memory_type": "semantic",
                    "importance": 0.9,
                    "source": "explicit_user_request",
                }]
                mem = memory_service.add_memory(
                    user_id=state["user_id"],
                    content=memory_content,
                    memory_type="semantic",
                    importance=0.9,
                    metadata={"source_type": "explicit_user_request", "source": "explicit_user_request",
                              "raw_user_input": user_input, "explicit": True, "run_id": str(state["run_id"])},
                    db=self.db,
                )
                # ── Build structured MemorySaveResult ──
                write_ok = mem.get("ok", False)
                save_result = {
                    "ok": write_ok,
                    "memory_id": mem.get("id"),
                    "qdrant_point_id": mem.get("qdrant_point_id"),
                    "memory_type": "semantic",
                    "content": memory_content,
                    "category": mem.get("category", ""),
                    "status": "active",
                    "qdrant_indexed": mem.get("qdrant_indexed", False),
                    "error": mem.get("error"),
                    "deduped": mem.get("deduped", False),
                    "updated_existing": mem.get("updated_existing", False),
                }
                state.setdefault("memory_save_results", []).append(save_result)

                # Only append to memory_updates if write actually succeeded
                if write_ok:
                    state.setdefault("memory_updates", []).append(mem)
                    state["memory_write_result"] = {
                        "success": True,
                        "content": memory_content,
                        "memory_type": "semantic",
                        "memory_id": mem.get("id"),
                        "qdrant_indexed": mem.get("qdrant_indexed", False),
                    }
                    state["memory_result"] = {"saved_count": 1, "semantic": 1, "episodic": 0}
                    # ── ONLY set final_output if write CONFIRMED ok ──
                    if mem.get("qdrant_indexed"):
                        state["final_output"] = f"已记住：{memory_content}"
                    else:
                        state["final_output"] = f"已记录：{memory_content}（向量索引暂不可用，搜索可能受限）"
                else:
                    # Write failed — do NOT claim success
                    append_error(state, "memory_agent",
                                 f"Memory write failed: {mem.get('error', 'unknown')}")
                    state["memory_write_result"] = {
                        "success": False,
                        "content": memory_content,
                        "error": mem.get("error", "unknown"),
                    }
                    # Do NOT set final_output — let final_response handle it
                state["memory_write_decision"] = {
                    "should_write": bool(write_ok),
                    "mode": "explicit" if write_ok else "skipped",
                    "candidates_count": 1,
                    "accepted_count": 1 if write_ok else 0,
                    "rejected_count": 0 if write_ok else 1,
                    "accepted_memory_ids": [mem.get("id")] if write_ok and mem.get("id") else [],
                    "reasons": ["explicit_user_request"] if write_ok else [str(mem.get("error") or "memory_write_failed")],
                    "thresholds": thresholds,
                    "rules_matched": ["explicit_memory_write"],
                }
                append_pipeline_step(
                    state,
                    "memory_writer",
                    detail="explicit memory write processed",
                    extra=state["memory_write_decision"],
                )

                append_output(state, "memory_agent", state.get("memory_write_result", {}))
                append_agent_result(state, AgentResult(
                    task_id=task_id_for_agent(state, "memory_agent"),
                    agent="memory_agent",
                    status="ok" if write_ok else "failed",
                    confidence=0.9 if write_ok else 0.0,
                    summary=f"Memory recorded: {memory_content}" if write_ok else "Memory write failed",
                    findings=[memory_content] if write_ok else [],
                    memory_updates=[mem] if write_ok else [],
                    errors=[] if write_ok else [str(mem.get("error") or "unknown")],
                    warnings=[] if write_ok else ["memory_write_failed"],
                ))
                record_step(self.db, state["run_id"], "memory_agent", "write_memory_explicit",
                            {"user_input": user_input},
                            {"memory_write_result": state["memory_write_result"]})
                append_status_step(
                    state,
                    key="memory_agent",
                    node_name="memory_agent",
                    detail=f"已明确写入 semantic 记忆 1 条, ok={write_ok}",
                    model=resolve_model_name("memory", complexity="low").model,
                    extra={"memory_writes": 1, "explicit": True, "ok": write_ok},
                )
                emit_visible_thought(self.db, state, "memory_agent")
                mark_completed(state, "memory_agent")
                record_agent_node_result(
                    state,
                    node="memory_agent",
                    updates={
                        "memory_result": state.get("memory_result"),
                        "memory_write_result": state.get("memory_write_result"),
                        "memory_write_decision": state.get("memory_write_decision", {}),
                        "memory_save_results": state.get("memory_save_results", []),
                        "memory_updates": state.get("memory_updates", []),
                    },
                    summary=(state.get("memory_write_result") or {}).get("content", ""),
                )
                return state

            # Only write memory for high-value tasks, not casual chat
            if intent in ("chat",) or not bool(self.payload.get("write_memory", False)):
                state["memory_write_decision"] = {
                    "should_write": False,
                    "mode": "skipped",
                    "candidates_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "accepted_memory_ids": [],
                    "reasons": ["memory_writer_not_requested"],
                    "thresholds": thresholds,
                    "rules_matched": [],
                }
                append_pipeline_step(
                    state,
                    "memory_writer",
                    status="skipped",
                    detail="memory writer skipped",
                    extra=state["memory_write_decision"],
                )
                mark_completed(state, "memory_agent")
                record_agent_node_result(
                    state,
                    node="memory_agent",
                    updates={"memory_write_decision": state["memory_write_decision"]},
                    status="skipped",
                    summary="Skipped memory extraction because memory_writer was not requested.",
                )
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

            result = await memory_service.async_extract_and_save(
                user_id=state["user_id"], user_input=user_input,
                agent_output=agent_output, page_context=page_context,
                feed_card_context=feed_card_context,
                matched_skill=matched_skill,
                created_skill_draft=created_skill_draft,
                db=self.db,
                run_id=str(state["run_id"]),
                use_llm=True,
                thread_id=state.get("thread_id", ""),
            )
            saved = result.get("saved", {})
            all_saved = saved.get("working", []) + saved.get("episodic", []) + saved.get("semantic", [])
            extraction = result.get("extraction") or {}
            candidates_count = (
                len(extraction.get("working_memories", []) or [])
                + len(extraction.get("episodic_memories", []) or [])
                + len(extraction.get("semantic_memories", []) or [])
            )
            state.setdefault("memory_updates", []).extend(all_saved)
            # ── Populate memory_save_results from structured results ──
            save_results = result.get("save_results", [])
            state.setdefault("memory_save_results", []).extend(save_results)
            state["memory_result"] = {
                "saved_count": len(all_saved),
                "semantic": len(saved.get("semantic", [])),
                "episodic": len(saved.get("episodic", [])),
                "ok_count": sum(1 for r in save_results if r.get("ok")),
                "qdrant_indexed_count": sum(1 for r in save_results if r.get("qdrant_indexed")),
            }
            state["memory_write_decision"] = {
                "should_write": bool(all_saved),
                "mode": "auto_high_confidence" if all_saved else "implicit_candidate",
                "candidates_count": candidates_count,
                "accepted_count": len(all_saved),
                "rejected_count": max(0, candidates_count - len(all_saved)),
                "accepted_memory_ids": [item.get("id") for item in all_saved if item.get("id")],
                "reasons": ["high_confidence_auto_write"] if all_saved else ["below_write_threshold"],
                "thresholds": thresholds,
                "rules_matched": ["payload_write_memory_enabled"],
            }
            append_pipeline_step(
                state,
                "memory_writer",
                detail="memory writer completed",
                extra=state["memory_write_decision"],
            )
            append_output(state, "memory_agent", state["memory_result"])
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "memory_agent"),
                agent="memory_agent",
                status="ok",
                confidence=0.8 if state["memory_result"]["ok_count"] else 0.5,
                summary=f"Saved {state['memory_result']['saved_count']} memory items",
                findings=[f"saved_count={state['memory_result']['saved_count']}"],
                memory_updates=all_saved,
                warnings=[] if state["memory_result"]["ok_count"] else ["memory_not_indexed_or_not_saved"],
            ))
            record_step(self.db, state["run_id"], "memory_agent", "write_memory", {},
                        {"memory_result": state["memory_result"]})
            append_status_step(
                state,
                key="memory_agent",
                node_name="memory_agent",
                detail=f"写入记忆 {state['memory_result']['saved_count']} 条, ok={state['memory_result']['ok_count']}",
                model=resolve_model_name("memory", complexity="low").model,
                extra={"memory_writes": state["memory_result"]["saved_count"]},
            )
        except Exception as exc:
            append_error(state, "memory_agent", str(exc))
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "memory_agent"),
                agent="memory_agent",
                status="failed",
                confidence=0.0,
                summary="Memory agent failed",
                errors=[str(exc)],
                warnings=["memory_agent_failed"],
            ))
        emit_visible_thought(self.db, state, "memory_agent")
        mark_completed(state, "memory_agent")
        record_agent_node_result(
            state,
            node="memory_agent",
            updates={
                "memory_result": state.get("memory_result"),
                "memory_write_result": state.get("memory_write_result"),
                "memory_write_decision": state.get("memory_write_decision", {}),
                "memory_save_results": state.get("memory_save_results", []),
                "memory_updates": state.get("memory_updates", []),
            },
            summary=(state.get("memory_result") or {}).get("summary", ""),
        )
        return state

    async def skill_agent(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Skill Agent: detect reusable workflows and optionally create skill drafts."""
        if state.get("route") in {"approval", "blocked"}:
            mark_completed(state, "skill_agent")
            record_agent_node_result(
                state,
                node="skill_agent",
                updates={},
                status="skipped",
                summary="Skipped because route is approval or blocked.",
            )
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "skill_agent"),
                agent="skill_agent",
                status="ok",
                confidence=float(reuse.get("reusable_score", 0) or 0),
                summary=state["skill_result"].get("reason", ""),
                artifacts=[created] if created else [],
                findings=[f"reusable_score={state['skill_result']['reusable_score']}"],
            ))
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
            append_agent_result(state, AgentResult(
                task_id=task_id_for_agent(state, "skill_agent"),
                agent="skill_agent",
                status="failed",
                confidence=0.0,
                summary="Skill agent failed.",
                errors=[str(exc)],
                warnings=["skill_agent_failed"],
            ))
        emit_visible_thought(self.db, state, "skill_agent")
        mark_completed(state, "skill_agent")
        record_agent_node_result(
            state,
            node="skill_agent",
            updates={
                "skill_result": state.get("skill_result"),
                "skill_reuse": state.get("skill_reuse"),
                "created_skill_draft": state.get("created_skill_draft"),
                "skill_drafts": state.get("skill_drafts", []),
            },
            summary=(state.get("skill_result") or {}).get("reason", ""),
        )
        return state

    def _is_explicit_memory_write(self, user_input: str) -> bool:
        """Check whether the user is explicitly asking to remember something."""
        if not user_input:
            return False
        text = user_input.strip().lower()
        explicit_literals = (
            "记住", "帮我记", "记一下", "记录一下", "以后记得",
            "别忘了", "下次记住", "保存下来", "写入记忆", "存入记忆",
            "长期记忆", "以后都", "以后要", "以后默认",
            "remember", "save preference", "save my preference",
            "don't forget", "do not forget",
        )
        return any(pattern in text for pattern in self._MEMORY_WRITE_PATTERNS) or any(
            literal in text for literal in explicit_literals
        )

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



