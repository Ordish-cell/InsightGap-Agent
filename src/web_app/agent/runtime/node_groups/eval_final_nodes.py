from __future__ import annotations

from src.web_app.agent.runtime.node_groups.base import *
from src.web_app.agent.runtime.contracts import build_runtime_contract_report
from src.web_app.agent.runtime.latency import build_runtime_latency_trace
from src.web_app.agent.runtime.replanner import (
    build_replanner_candidate_plan,
    build_replanner_control_decision,
    build_replanner_shadow_report,
    update_replanner_shadow_metrics,
)
from src.web_app.agent.runtime.schemas import StateDelta
from src.web_app.agent.runtime.state_delta import apply_state_delta, record_agent_node_result, record_node_result


def _web_search_result_block(tool_result: dict[str, Any] | None) -> str:
    if not isinstance(tool_result, dict):
        return ""
    output = tool_result.get("output") if isinstance(tool_result.get("output"), dict) else tool_result
    tool_name = str(tool_result.get("tool_name") or output.get("tool_name") or "")
    if tool_name != "web.search" and "results" not in output:
        return ""
    query = str(output.get("query") or "")
    final_query = str(output.get("final_query") or query)
    provider = str(output.get("provider") or "")
    error = str(output.get("error") or "")
    results = output.get("results") if isinstance(output.get("results"), list) else []
    rounds = output.get("search_rounds") if isinstance(output.get("search_rounds"), list) else []
    reasoning_summary = str(output.get("reasoning_summary") or "")
    lines = [
        "[Web Search Results]",
        f"Query: {query}",
        f"Final Query: {final_query}",
        f"Provider: {provider or 'unavailable'}",
        "Use these results only for this answer. Cite source URLs when making claims from search.",
        "For latest/current questions, do not use stale model knowledge or make dated claims that are not supported by these results.",
    ]
    if reasoning_summary:
        lines.append(f"Search Summary: {reasoning_summary}")
    for item in rounds[:2]:
        if isinstance(item, dict):
            lines.append(
                f"Round {item.get('round')}: query={item.get('query')} "
                f"results={item.get('result_count')} observation={item.get('observation')}"
            )
    if not results:
        lines.append(f"Search failed or returned no results: {error or 'no_results'}")
        lines.append("Do not answer latest/current facts from memory when search returned no results. State that live verification failed.")
        return "\n".join(lines)
    for index, item in enumerate(results[:8], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("url") or f"Result {index}")
        url = str(item.get("url") or "")
        snippet = str(item.get("snippet") or "")[:500]
        published_at = str(item.get("published_at") or "")
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if published_at:
            lines.append(f"   Published: {published_at}")
        if snippet:
            lines.append(f"   Summary: {snippet}")
    return "\n".join(lines)


def _local_tool_result_block(tool_result: dict[str, Any] | None) -> str:
    if not isinstance(tool_result, dict):
        return ""
    output = tool_result.get("output") if isinstance(tool_result.get("output"), dict) else tool_result
    tool_name = str(tool_result.get("tool_name") or output.get("tool_name") or "")
    if not tool_name.startswith("system."):
        return ""
    safe_output = {
        key: value
        for key, value in output.items()
        if not str(key).startswith("_")
    }
    try:
        output_text = json.dumps(safe_output, ensure_ascii=False, default=str)
    except TypeError:
        output_text = str(safe_output)
    return (
        "[Local Tool Result]\n"
        f"Tool: {tool_name}\n"
        f"Output: {output_text}\n"
        "Use this local tool output directly. Do not call or mention web search for this result."
    )


def _fallback_local_tool_answer(tool_result: dict[str, Any] | None) -> str:
    if not isinstance(tool_result, dict):
        return ""
    output = tool_result.get("output") if isinstance(tool_result.get("output"), dict) else {}
    tool_name = str(tool_result.get("tool_name") or output.get("tool_name") or "")
    if not tool_name.startswith("system.") or not isinstance(output, dict):
        return ""
    if tool_name == "system.time":
        date = output.get("date", "")
        time_value = output.get("time", "")
        weekday = output.get("weekday_zh") or output.get("weekday", "")
        timezone = output.get("timezone", "")
        return f"当前日期是 {date}，{weekday}。当前时间是 {time_value}（{timezone}）。"
    if tool_name == "system.calc":
        return f"计算结果是：{output.get('result')}"
    if tool_name == "system.unit_convert":
        return f"换算结果是：{output.get('result')} {output.get('to', '')}".strip()
    if tool_name == "system.uuid":
        return f"生成的 UUID 是：{output.get('uuid')}"
    if tool_name == "system.hash":
        return f"{output.get('algorithm', 'hash')} 结果是：{output.get('hash')}"
    return ""


class EvalFinalNodesMixin:
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
        warnings: list[str] = []
        constraints: list[str] = []
        route_plan = state.get("route_plan") or {}
        route_agents = set(route_plan.get("route", []) or [])
        formal_agents = {
            "rag_agent",
            "memory_agent",
            "tool_agent",
            "artifact_agent",
            "skill_agent",
            "research_agent",
        }
        prefetch_agents = {"rag_prefetch", "memory_prefetch", "skill_prefetch", "graph_prefetch"}

        def _add_warning(value: str) -> None:
            if value and value not in warnings:
                warnings.append(value)

        def _add_constraint(value: str) -> None:
            if value and value not in constraints:
                constraints.append(value)

        def _is_formal_result(result: dict[str, Any]) -> bool:
            agent = str(result.get("agent") or "")
            metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
            if agent in prefetch_agents or agent.endswith("_prefetch"):
                return False
            if metadata.get("source") == "prefetch":
                return False
            return agent in formal_agents

        formal_results = [
            result for result in list(state.get("agent_results") or [])
            if isinstance(result, dict) and _is_formal_result(result)
        ]

        def _latest_agent_result(agent: str) -> dict[str, Any] | None:
            for result in reversed(formal_results):
                if result.get("agent") == agent:
                    return result
            return None

        tool_result = _latest_agent_result("tool_agent")
        if tool_result:
            tool_status = str(tool_result.get("status") or "")
            if tool_status == "needs_approval":
                _add_warning("tool_waiting_approval")
                _add_constraint("Do not claim the tool action has been completed. Tell the user approval is required before execution.")
            elif tool_status in {"failed", "denied"}:
                _add_warning("tool_denied" if tool_status == "denied" else "tool_failed")
                _add_constraint("Do not claim the tool action has been completed.")
        elif "tool_agent" in route_agents:
            legacy_tool_result = state.get("tool_result") or state.get("tool_call") or {}
            legacy_tool_status = str(legacy_tool_result.get("status") or "")
            if legacy_tool_status in {"waiting_approval", "missing_fields"}:
                _add_warning("tool_waiting_approval")
                _add_constraint("Do not claim the tool action has been completed. Tell the user approval is required before execution.")
            elif legacy_tool_status in {"failed", "blocked", "rejected"}:
                _add_warning("tool_denied" if legacy_tool_status in {"blocked", "rejected"} else "tool_failed")
                _add_constraint("Do not claim the tool action has been completed.")

        artifact_planned = "artifact_agent" in route_agents or route_plan.get("intent") == "artifact"
        if artifact_planned:
            artifact_result = _latest_agent_result("artifact_agent")
            if artifact_result:
                has_artifact = bool(artifact_result.get("artifacts") or state.get("artifacts"))
                if artifact_result.get("status") != "ok" or not has_artifact:
                    _add_warning("artifact_missing")
                    _add_constraint("Do not claim an artifact or file was generated unless artifact_agent succeeded.")
            else:
                legacy_artifact_result = state.get("artifact_result") or {}
                if legacy_artifact_result.get("error") or not state.get("artifacts"):
                    _add_warning("artifact_missing")
                    _add_constraint("Do not claim an artifact or file was generated unless artifact_agent succeeded.")

        memory_write = state.get("memory_write_result") or {}
        memory_planned = (
            "memory_agent" in route_agents
            or route_plan.get("intent") == "memory"
            or bool(memory_write)
        )
        memory_result = _latest_agent_result("memory_agent")
        if memory_planned:
            if memory_result and memory_result.get("status") != "ok":
                _add_warning("memory_write_failed")
                _add_constraint("Do not claim the memory was saved unless memory_agent succeeded.")
            elif memory_write and memory_write.get("success") is False:
                _add_warning("memory_write_failed")
                _add_constraint("Do not claim the memory was saved unless memory_agent succeeded.")

        rag_result = _latest_agent_result("rag_agent")
        if rag_result and rag_result.get("status") == "ok" and not rag_result.get("evidence"):
            _add_warning("rag_evidence_missing")
            _add_warning("evidence_missing")
            _add_constraint("Do not imply that the answer is backed by retrieved document evidence when no RAG evidence is available.")
        elif not rag_result and "rag_agent" in route_agents:
            legacy_rag = state.get("rag_result") or state.get("rag") or {}
            if legacy_rag and not legacy_rag.get("evidence"):
                _add_warning("rag_evidence_missing")
                _add_warning("evidence_missing")
                _add_constraint("Do not imply that the answer is backed by retrieved document evidence when no RAG evidence is available.")

        if "research_agent" in route_agents and route_plan.get("explicit_research") is False:
            _add_warning("research_fallback_mode")
        if status == "waiting_approval":
            _add_warning("approval_pending")
            _add_constraint("Do not claim the tool action has been completed. Tell the user approval is required before execution.")
        evaluation_result = EvaluationResult(
            pass_=not constraints,
            score=1.0 if not constraints else 0.65,
            missing=warnings,
            final_response_constraints=constraints,
            warnings=warnings,
        )
        evaluation = dict(state["evaluation"])
        evaluation["warnings"] = warnings
        evaluation["constraints"] = constraints
        evaluation["score"] = evaluation_result.score
        delta = StateDelta(
            updates={
                "evaluation_result": evaluation_result.model_dump(),
                "final_response_constraints": constraints,
                "final_warnings": warnings,
                "evaluation": evaluation,
            },
            warnings=warnings,
            metadata={"source": "evaluator"},
        )
        apply_state_delta(state, delta)
        if not state.get("final_output") and status == "completed":
            state["final_output"] = "Agent runtime completed."
        append_status_step(
            state,
            key="evaluator",
            node_name="evaluator",
            detail=f"评估完成，状态 {status}",
            extra={"status": status, "errors": len(state.get("errors", [])), "warnings": warnings},
        )
        emit_visible_thought(self.db, state, "evaluator", stream_queue=self._stream_queue)
        record_step(self.db, state["run_id"], "evaluator", "evaluate", {"route": state.get("route")}, {"evaluation": state["evaluation"]})
        mark_completed(state, "evaluator")
        record_node_result(
            state,
            node="evaluator",
            delta=delta,
            summary="Evaluated formal agent outputs.",
        )
        return state

    # ── Multi-Agent Supervisor nodes ──────────────────────────────────

    async def final_response(self, state: AgentRuntimeState) -> AgentRuntimeState:
        """Final Response: use the configured final LLM to write the user-facing answer.

        Only reached when all route nodes have completed (or on reject/resume).
        Never reached during waiting_approval — the graph interrupts to END before
        reaching this node.
        """
        route_plan = state.get("route_plan") or {}
        intent = route_plan.get("intent", "chat")
        answer_mode = route_plan.get("answer_mode") or state.get("answer_mode") or "chat"
        is_conversation_recall = answer_mode == "conversation_recall"

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
            emit_visible_thought(self.db, state, "final_response", stream_queue=self._stream_queue)
            await self._enrich_visible_thoughts_with_llm(state)

        draft_answer = "\n\n".join(answer_parts)
        used_streaming_llm = False
        append_pipeline_step(state, "generate_answer", detail="calling final answer model")
        if is_conversation_recall:
            final_answer = await self._generate_final_answer_with_llm(state, draft_answer)
            used_streaming_llm = True
            if self._conversation_recall_has_history(state) and self._looks_like_no_history_claim(final_answer):
                state["_conversation_recall_retry"] = True
                final_answer = await self._generate_final_answer_with_llm(state, draft_answer)
                if self._looks_like_no_history_claim(final_answer):
                    final_answer = self._fallback_final_answer(state, draft_answer)
        elif intent in ("document_qa",):
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

        # ── Memory-confirmation guard ──
        # Only allow '已记住' when memory_save_results has ok=True entries.
        # Never trust memory_candidates or LLM self-inference.
        final_answer = self._sanitize_memory_claims(
            final_answer,
            state.get("memory_save_results", []),
            state.get("memory_write_result") or {},
        )

        state["final_answer"] = final_answer
        state["final_output"] = final_answer
        append_agent_result(state, AgentResult(
            task_id=task_id_for_agent(state, "final_response"),
            agent="final_response",
            status="ok" if final_answer else "failed",
            confidence=0.9 if final_answer else 0.0,
            summary=final_answer,
            errors=[] if final_answer else ["final_answer_empty"],
            warnings=list(state.get("final_warnings") or []),
        ))

        errors = state.get("errors", [])
        status = state.get("status") or ("failed" if errors else "completed")
        latency = build_runtime_latency_trace(state)
        state["runtime_latency_trace"] = latency["runtime_latency_trace"]
        state["runtime_latency_warnings"] = latency["runtime_latency_warnings"]
        state["runtime_slow_path_hints"] = latency["runtime_slow_path_hints"]

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
            "evaluation_result": state.get("evaluation_result", {}),
            "final_response_constraints": state.get("final_response_constraints", []),
            "final_warnings": state.get("final_warnings", []),
            "errors": errors,
            "agent_outputs": state.get("agent_outputs", []),
            "agent_results": state.get("agent_results", []),
            "prefetch_results": state.get("prefetch_results", {}),
            "prefetch_warnings": state.get("prefetch_warnings", []),
            "prefetch_elapsed_ms": state.get("prefetch_elapsed_ms", 0),
            "parallel_read_results": state.get("parallel_read_results", {}),
            "parallel_read_warnings": state.get("parallel_read_warnings", []),
            "parallel_read_elapsed_ms": state.get("parallel_read_elapsed_ms", 0),
            "supervisor_decision": state.get("supervisor_decision", {}),
            "supervisor_warnings": state.get("supervisor_warnings", []),
            "supervisor_trace": state.get("supervisor_trace", []),
            "supervisor_dispatch_audit": state.get("supervisor_dispatch_audit", {}),
            "supervisor_dispatch_warnings": state.get("supervisor_dispatch_warnings", []),
            "supervisor_shadow_policy": state.get("supervisor_shadow_policy", {}),
            "supervisor_policy_warnings": state.get("supervisor_policy_warnings", []),
            "supervisor_shadow_metrics": state.get("supervisor_shadow_metrics", {}),
            "supervisor_readiness_report": state.get("supervisor_readiness_report", {}),
            "supervisor_readiness_warnings": state.get("supervisor_readiness_warnings", []),
            "supervisor_control_decision": state.get("supervisor_control_decision", {}),
            "supervisor_control_warnings": state.get("supervisor_control_warnings", []),
            "runtime_latency_trace": state.get("runtime_latency_trace", {}),
            "runtime_latency_warnings": state.get("runtime_latency_warnings", []),
            "runtime_slow_path_hints": state.get("runtime_slow_path_hints", []),
            "replanner_shadow_metrics": state.get("replanner_shadow_metrics", {}),
            "replanner_control_decision": state.get("replanner_control_decision", {}),
            "replanner_control_warnings": state.get("replanner_control_warnings", []),
            "memory_context": state.get("memory_context", {}),
            "memory_write_decision": state.get("memory_write_decision", {}),
            "conversation_recall_context": state.get("conversation_recall_context", {}),
            "pipeline_steps": state.get("pipeline_steps", []),
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
        append_pipeline_step(state, "finalize", detail="final payload ready")
        final_payload["pipeline_steps"] = state.get("pipeline_steps", [])
        state["final_payload"] = final_payload
        state["status"] = status
        record_agent_node_result(
            state,
            node="final_response",
            updates={
                "final_answer": state.get("final_answer"),
                "final_output": state.get("final_output"),
                "final_payload": state.get("final_payload"),
                "status": state.get("status"),
                "runtime_latency_trace": state.get("runtime_latency_trace", {}),
            },
            summary=final_answer,
        )
        contract = build_runtime_contract_report(state)
        state["runtime_contract_report"] = contract["runtime_contract_report"]
        state["runtime_contract_warnings"] = contract["runtime_contract_warnings"]
        replanner = build_replanner_shadow_report(state)
        state["replanner_shadow_report"] = replanner["replanner_shadow_report"]
        state["replanner_shadow_warnings"] = replanner["replanner_shadow_warnings"]
        final_payload["replanner_shadow_report"] = state["replanner_shadow_report"]
        final_payload["replanner_shadow_warnings"] = state["replanner_shadow_warnings"]
        candidate = build_replanner_candidate_plan(state)
        state["replanner_candidate_plan"] = candidate["replanner_candidate_plan"]
        state["replanner_candidate_warnings"] = candidate["replanner_candidate_warnings"]
        final_payload["replanner_candidate_plan"] = state["replanner_candidate_plan"]
        final_payload["replanner_candidate_warnings"] = state["replanner_candidate_warnings"]
        replanner_control = build_replanner_control_decision(
            state,
            (state.get("replanner_control_decision") or {}).get("legacy_next_node") or "final_response",
        )
        state["replanner_control_decision"] = replanner_control["replanner_control_decision"]
        state["replanner_control_warnings"] = replanner_control["replanner_control_warnings"]
        replanner_observation = {
            **replanner,
            **candidate,
            **replanner_control,
        }
        state["replanner_shadow_metrics"] = update_replanner_shadow_metrics(state, replanner_observation)
        final_payload["replanner_shadow_metrics"] = state["replanner_shadow_metrics"]
        final_payload["replanner_control_decision"] = state["replanner_control_decision"]
        final_payload["replanner_control_warnings"] = state["replanner_control_warnings"]
        contract = build_runtime_contract_report(state)
        state["runtime_contract_report"] = contract["runtime_contract_report"]
        state["runtime_contract_warnings"] = contract["runtime_contract_warnings"]
        final_payload["runtime_contract_report"] = state["runtime_contract_report"]
        final_payload["runtime_contract_warnings"] = state["runtime_contract_warnings"]
        state["final_payload"] = final_payload
        if state.get("node_results"):
            state["node_results"][-1].setdefault("delta", {}).setdefault("updates", {})["final_payload"] = final_payload

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
        queue = getattr(self, "_stream_queue", None) or state.get("_stream_queue")
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
        if (route_plan.get("answer_mode") or state.get("answer_mode")) == "conversation_recall":
            return self._build_conversation_recall_prompt(state)

        # ── Build the core prompt ───────────────────────────────────
        # When GSSC context is available, it becomes the primary context.
        # When it's empty, fall back to a legacy flat payload.
        if gssc_context:
            return self._build_gssc_prompt(state, gssc_context, draft_answer, route_plan)
        return self._build_legacy_prompt(state, draft_answer, route_plan)

    def _build_conversation_recall_prompt(self, state: AgentRuntimeState) -> str:
        recall_context = state.get("conversation_recall_context") or {}
        previous = list(recall_context.get("previous_user_messages") or [])
        messages = list(recall_context.get("messages") or [])
        if not messages:
            history_text = str((state.get("context") or {}).get("conversation_history") or "")
            messages = [{"role": "history", "content": history_text}] if history_text else []
        previous_block = "\n".join(f"- {item}" for item in previous[-12:])
        messages_block = "\n".join(
            f"{item.get('role', 'message')}: {item.get('content', '')}"
            for item in messages[-12:]
            if isinstance(item, dict)
        )
        retry_line = (
            "You previously claimed there was no conversation history. That is incorrect: use the provided messages.\n"
            if state.get("_conversation_recall_retry")
            else ""
        )
        return (
            "You are the final answer node. Answer in Chinese unless the user asks otherwise.\n"
            "Hard rule: this is conversation_recall. Use ONLY the current AgentConversation/AgentMessage history below.\n"
            "Do not use long-term memory, semantic memory, episodic memory, Qdrant, PG memory fallback, RAG, feed cards, or profile memory.\n"
            "If previous user messages exist, never say you cannot access this conversation or that every conversation is independent.\n"
            f"{retry_line}\n"
            f"[Current User Input]\n{state.get('user_input', '')}\n\n"
            f"[Previous User Messages]\n{previous_block or '(none)'}\n\n"
            f"[Recent Conversation Messages]\n{messages_block or '(none)'}\n\n"
            "Task: briefly answer what the user asked before. If the user asks to analyze intent, summarize their likely intent from the previous user messages. "
            "Keep it grounded and do not invent messages."
        )

    @staticmethod
    def _conversation_recall_has_history(state: AgentRuntimeState) -> bool:
        recall_context = state.get("conversation_recall_context") or {}
        if recall_context.get("previous_user_messages"):
            return True
        history = str((state.get("context") or {}).get("conversation_history") or "")
        return "User:" in history

    @staticmethod
    def _looks_like_no_history_claim(answer: str) -> bool:
        text = (answer or "").lower()
        blockers = (
            "没有历史", "没有这个会话", "不能访问之前", "无法访问之前",
            "无法读取之前", "每次对话都是独立", "不保存历史", "不记得你之前",
            "no history", "can't access previous", "cannot access previous",
            "each conversation is independent",
        )
        return any(item in text for item in blockers)

    def _final_response_constraint_block(self, state: AgentRuntimeState) -> str:
        constraints = [str(item) for item in (state.get("final_response_constraints") or []) if item]
        warnings = [str(item) for item in (state.get("final_warnings") or []) if item]
        if not constraints and not warnings:
            return ""
        lines = [
            "[Runtime Safety Constraints]",
            "These constraints are authoritative. Do not contradict them in the final answer.",
        ]
        lines.extend(f"- {item}" for item in constraints)
        if warnings:
            lines.append("Warnings: " + ", ".join(warnings))
        return "\n".join(lines)

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
        constraint_block = self._final_response_constraint_block(state)
        if constraint_block:
            extra_blocks.append(constraint_block)
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
                "\n".join(str(a.get("title") or a.get("id", "")) for a in artifacts[:5])
            )
        web_search_block = _web_search_result_block(tool_result)
        if web_search_block:
            extra_blocks.append(web_search_block)
        local_tool_block = _local_tool_result_block(tool_result)
        if local_tool_block:
            extra_blocks.append(local_tool_block)
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
        elif intent == "tool.web_search":
            system_instruction += "8. 如果 [Web Search Results] 有结果，必须基于搜索结果回答并引用来源 URL；不要用旧知识覆盖搜索结果。若搜索失败或 results=0，必须明确说未能完成联网验证，不要编造“最新/当前”事实。\n"
        elif str(intent).startswith("system."):
            system_instruction += "8. 如果 [Local Tool Result] 有结果，请直接基于本地工具输出回答；不要说需要联网，不要编造权限限制。\n"
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
        web_search_block = _web_search_result_block(payload.get("tool_result") or {})
        local_tool_block = _local_tool_result_block(payload.get("tool_result") or {})
        constraint_block = self._final_response_constraint_block(state)
        runtime_context = (
            (f"{constraint_block}\n" if constraint_block else "") +
            f"意图: {payload.get('intent', 'chat')} | 风险: {payload.get('risk_level', 'L0')}\n"
            + (f"会话摘要: {context_summary}\n" if context_summary else "")
            + (f"关联信息流: {feed_title}\n" if feed_title else "")
            + (f"研究摘要: {research_summary[:300]}\n" if research_summary else "")
            + (f"RAG 回答: {rag_answer[:300]}\n" if rag_answer else "")
            + (f"工具状态: {tool_status}\n" if tool_status else "")
            + (f"{web_search_block}\n" if web_search_block else "")
            + (f"{local_tool_block}\n" if local_tool_block else "")
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
        if (route_plan.get("answer_mode") or state.get("answer_mode")) == "conversation_recall":
            recall_context = state.get("conversation_recall_context") or {}
            previous = [str(item) for item in (recall_context.get("previous_user_messages") or []) if item]
            if previous:
                bullets = "\n".join(f"- {item}" for item in previous[-8:])
                return f"当前会话历史里，你前面问过这些：\n\n{bullets}"
            return "当前会话历史通道是打开的，但我没有找到这条消息之前的用户提问。"
        local_tool_answer = _fallback_local_tool_answer(state.get("tool_result") or state.get("tool_call") or {})
        if local_tool_answer:
            return local_tool_answer
        # ── Memory write: confirm the save ONLY if actually written ──
        mem_result = state.get("memory_write_result") or {}
        save_results = state.get("memory_save_results", [])
        has_ok = any(r.get("ok") for r in save_results)
        if mem_result.get("success") and has_ok:
            content = mem_result.get("content", "")
            return f"已记住：{content}"
        if mem_result.get("success") is False:
            return f"未能保存：{mem_result.get('error', '写入失败')}"
        if (intent == "memory" or self._is_explicit_memory_write(user_input)) and has_ok:
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


    @staticmethod
    def _sanitize_memory_claims(answer: str, save_results: list[dict[str, Any]],
                                 memory_write_result: dict[str, Any]) -> str:
        """Remove or correct false memory-save claims from the final answer.

        Hard rule: '已记住' / '已记录' is ONLY allowed when at least one
        entry in memory_save_results has ok=True.  Never trust memory_candidates
        or LLM self-inference.
        """
        has_ok = any(r.get("ok") for r in save_results)
        explicit_failed = memory_write_result.get("success") is False

        if explicit_failed:
            # Explicit memory write failed — strip false claims
            import re as _re
            answer = _re.sub(r'已记住[：:]\s*[^\n。.]+', '未能保存该信息，请稍后重试', answer)
            answer = _re.sub(r'已记录[：:]\s*[^\n。.]+', '未能保存该信息，请稍后重试', answer)
            answer = answer.replace("已记住", "未能保存")
            answer = answer.replace("已记录", "未能保存")
            answer = answer.replace("已保存", "未能保存")
            return answer

        if not has_ok and save_results:
            # Candidates exist but none confirmed ok
            return (
                "我识别到这可能是长期记忆，但当前没有确认写入成功，"
                "所以不能说已经记住。请检查记忆写入链路。"
            )

        if has_ok and not any(r.get("qdrant_indexed") for r in save_results if r.get("ok")):
            # PG succeeded but Qdrant failed — append note to answer
            if "已记住" in answer or "已记录" in answer:
                if "向量索引暂不可用" not in answer and "搜索可能受限" not in answer:
                    answer += "\n\n（注：向量索引暂不可用，语义搜索可能受限。）"

        return answer


