"""Shared truthful answer projection, with no routing or model decisions."""

from src.web_app.agent.runtime.event_ledger import publish_event


def recover_answer(nodes, state):
    """The ledger survives a crash between a streamed answer and its checkpoint."""
    if nodes.db is None or state.get("_answer_started_emitted"):
        return None
    from sqlalchemy import select
    from src.web_app.models.orm import AgentEvent

    rows = (
        nodes.db.execute(
            select(AgentEvent)
            .where(
                AgentEvent.run_id == state["run_id"],
                AgentEvent.user_id == state["user_id"],
                AgentEvent.event_type.in_(
                    ("answer_started", "answer_delta", "answer_completed")
                ),
            )
            .order_by(AgentEvent.id)
        )
        .scalars()
        .all()
    )
    if not rows:
        return None
    answer, completed = "", False
    for row in rows:
        payload = row.payload_json or {}
        if row.event_type == "answer_delta":
            answer += payload.get("text", "")
            state["_answer_delta_emitted"] = True
        elif row.event_type == "answer_completed":
            answer = payload.get("answer", answer)
            completed = True
            state["_answer_completed_emitted"] = True
            if payload.get("failed"):
                state["error"] = "answer_generation_failed"
    state["_answer_started_emitted"] = True
    if not completed:
        state["error"] = "answer_stream_interrupted"
        state.setdefault("final_warnings", []).append(
            "回答流被中断，已保留可见内容；本轮不会重新生成整段答案。"
        )
    return finish(nodes, state, answer, failed=bool(state.get("error")))


def emit(nodes, state, kind, payload):
    if kind.startswith("answer_") and state.get("native_final_text_id"):
        payload = {**payload, "text_id": state["native_final_text_id"]}
    if kind.startswith("tool_call_"):
        payload = dict(payload)
        for snake, camel in (("tool_call_id", "toolCallId"), ("tool_name", "toolName"),
                             ("args_preview", "argsPreview"), ("output_preview", "outputPreview")):
            if snake in payload:
                payload[camel] = payload[snake]
    if nodes.db is not None:
        publish_event(
            nodes.db,
            nodes._stream_queue,
            state["run_id"],
            kind,
            payload,
            node_name="supervisor",
            user_id=state["user_id"],
            thread_id=state.get("thread_id"),
        )


def finish(nodes, state, answer, *, failed=False):
    from .policy import public_result

    if not state.get("_answer_started_emitted"):
        emit(nodes, state, "answer_started", {})
        state["_answer_started_emitted"] = True
    if not state.get("_answer_delta_emitted") and answer:
        emit(nodes, state, "answer_delta", {"text": answer})
        state["_answer_delta_emitted"] = True
    if not state.get("_answer_completed_emitted"):
        emit(nodes, state, "answer_completed", {"answer": answer, "failed": failed})
        state["_answer_completed_emitted"] = True
    state.update(
        final_answer=answer,
        final_output=answer,
        status="failed" if failed else "completed",
    )
    state["final_payload"] = {
        "answer": answer,
        "runtime_version": 2,
        "run_id": str(state["run_id"]),
        "thread_id": state.get("thread_id"),
        "conversation_id": state.get("conversation_id"),
        "research": state.get("research_result", {}),
        "rag": state.get("rag_result", {}),
        "artifacts": state.get("artifacts", []),
        "tool_calls": public_result(state.get("tool_calls", [])),
        "memory_writes": state.get("memory_updates", []),
        "skill_drafts": state.get("skill_drafts", []),
        "research_proposal": state.get("research_proposal"),
        "runtime_budget": state.get("runtime_budget", {}),
        "termination_reason": state.get("termination_reason", ""),
        "final_warnings": state.get("final_warnings", []),
        "errors": state.get("errors", []),
        "visible_thoughts": [],
        "route": [r["capability"] for r in state.get("observations", [])],
        "evaluation": state.get("evaluation", {}),
        "capability_results": public_result(state.get("observations", [])),
    }
    return state


def quality_constraints(state):
    constraints = [
        "Use only actual results. Never claim a tool ran, a file was saved, or memory was written without a successful receipt.",
        "A mock email with sent=false was NOT sent. No evidence means no retrieved support. Do not expose internal reasoning.",
    ]
    for result in state.get("observations", []):
        if result["status"] not in {"ok", "degraded"}:
            constraints.append(
                f"{result['action_id']}: {result['capability']} returned {result['status']}; do not claim success."
            )
    return "\n".join(constraints)
