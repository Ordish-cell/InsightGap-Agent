from __future__ import annotations

from src.web_app.agent.runtime.node_groups.base import *


class SetupNodesMixin:
    async def permission_guard(self, state: AgentRuntimeState) -> AgentRuntimeState:
        text = state.get("user_input", "")
        permission_level = "L0_READ_ONLY"
        if any(term in text for term in HIGH_RISK_TERMS) or any(term in text for term in ("删除", "支付", "付款", "转账")):
            permission_level = L4_HIGH_RISK
        elif any(term in text for term in EXTERNAL_WRITE_TERMS) or any(term in text for term in ("发邮件", "发送邮件", "邮件", "评论", "发布", "提交表单")) or any(re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE) for kw in _EN_EXTERNAL_WRITE_KEYWORDS):
            permission_level = L3_EXTERNAL_WRITE

        decision = PermissionGuard().check_tool_call("agent_runtime_task", permission_level)
        state["permission"] = {"level": permission_level, **decision}
        if permission_level == L4_HIGH_RISK:
            state["permission"]["requires_approval"] = True
            state["permission"]["reason"] = "strong_approval_required"
        record_step(self.db, state["run_id"], "permission_guard", "permission", {"user_input": text}, {"permission": state["permission"], "route": state.get("route")})
        emit_visible_thought(self.db, state, "permission_guard", stream_queue=self._stream_queue)
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
        print(f"[HOME_INTENT_TRACE] input={user_input[:200]} intent={home_intent.get('intent') or home_intent.get('detected_intent')} answer_mode={home_intent.get('answer_mode')} route_hints={home_intent.get('suggested_route_hints') or home_intent.get('required_agents')} source={home_intent.get('raw_intent_source', 'rule')}")
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
        emit_visible_thought(self.db, state, "home_intent_react", stream_queue=self._stream_queue)
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
        print(f"[PLANNER_TRACE] input={user_input[:200]} intent={route_plan.get('intent')} route={route_plan.get('route')} risk={route_plan.get('risk_level')} reasons={route_plan.get('reason')}")
        state["route_plan"] = route_plan
        state["execution_plan"] = execution_plan_from_route_plan(route_plan, state)
        route_intent = route_plan.get("intent", "chat")
        state["route"] = "tool" if str(route_intent).startswith("tool.") else route_intent  # legacy compat
        state["approval_required"] = route_plan.get("needs_approval", False)
        state["answer_mode"] = route_plan.get("answer_mode", "chat")
        append_pipeline_step(
            state,
            "understand_request",
            detail=f"intent={route_intent}; answer_mode={route_plan.get('answer_mode', 'chat')}",
            extra={
                "intent": route_intent,
                "answer_mode": route_plan.get("answer_mode", "chat"),
                "risk_level": route_plan.get("risk_level", "L0"),
            },
        )

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
        emit_visible_thought(self.db, state, "planner", stream_queue=self._stream_queue)
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
            answer_mode=route_plan.get("answer_mode", "chat"),
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


