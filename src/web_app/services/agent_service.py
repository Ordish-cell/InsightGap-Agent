import asyncio
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime import AgentRuntime
from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.db.repositories.agent_repository import AgentEventRepository, AgentRunRepository, AgentStepRepository


async def run_agent_async(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    user_input = payload.get("user_input") or payload.get("input") or payload.get("query") or ""
    payload = {**payload, "user_input": user_input}
    page_context = payload.get("page_context") or {}
    conversation_id = payload.get("conversation_id")
    selected_feed_card_id = page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
    if selected_feed_card_id and not payload.get("feed_card_id"):
        payload["feed_card_id"] = selected_feed_card_id
    run_repo = AgentRunRepository(db)
    run = run_repo.create(
        user_id=user_id,
        run_type=payload.get("run_type", "agent_runtime"),
        mode=payload.get("mode", "react"),
        status="running",
        user_input=user_input,
        graph_state={"source": payload.get("source", "agent_page"), "page_context": page_context},
    )
    thread_id = f"user:{user_id}:conversation:{conversation_id}" if conversation_id else f"user:{user_id}:run:{run.id}"
    run_repo.update(run, graph_state={"source": payload.get("source", "agent_page"), "page_context": page_context, "thread_id": thread_id, "conversation_id": conversation_id})
    record_event(db, run.id, "run_started", {"user_input": user_input, "source": payload.get("source", "agent_page")}, user_id=user_id, thread_id=thread_id)
    state = await AgentRuntime(db, payload).run({"user_id": user_id, "run_id": run.id, "thread_id": thread_id, "conversation_id": conversation_id, "user_input": user_input, "mode": run.mode, "source": payload.get("source", "agent_page"), "page_context": page_context})
    run_repo.update(run, status=state.get("status", "completed"), graph_state=_json_safe(dict(state)), result_summary=state.get("final_output", ""), error_message=state.get("error", ""))
    if state.get("approval_required") or state.get("status") == "waiting_approval":
        record_event(db, run.id, "approval_required", state.get("approval_payload") or {}, user_id=user_id, thread_id=thread_id)
    elif state.get("status") == "failed":
        record_event(db, run.id, "run_failed", {"status": state.get("status"), "final_output": state.get("final_output", "")}, user_id=user_id, thread_id=thread_id)
    else:
        record_event(db, run.id, "run_completed", {"status": state.get("status"), "final_output": state.get("final_output", "")}, user_id=user_id, thread_id=thread_id)
    return _run_response(run.id, state)


def run_agent(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(run_agent_async(db, user_id, payload))


def get_run(db: Session, user_id: int, run_id: int) -> dict[str, Any]:
    run = AgentRunRepository(db).get_by_user(user_id, run_id)
    if not run:
        raise ValueError("AgentRun not found")
    return {
        "id": run.id,
        "status": run.status,
        "run_type": run.run_type,
        "mode": run.mode,
        "user_input": run.user_input,
        "result_summary": run.result_summary,
        "error_message": run.error_message,
        "graph_state": run.graph_state or {},
        "thread_id": (run.graph_state or {}).get("thread_id", ""),
    }


def list_steps(db: Session, user_id: int, run_id: int) -> list[dict[str, Any]]:
    if not AgentRunRepository(db).get_by_user(user_id, run_id):
        raise ValueError("AgentRun not found")
    return [{"id": step.id, "node_name": step.node_name, "status": step.status, "input": step.input, "output": step.output} for step in AgentStepRepository(db).list_by_run(run_id)]


def list_events(db: Session, user_id: int, run_id: int) -> list[dict[str, Any]]:
    if not AgentRunRepository(db).get_by_user(user_id, run_id):
        raise ValueError("AgentRun not found")
    rows = AgentEventRepository(db).list_by_run(user_id, run_id)
    if rows:
        return [_event_to_sse(item) for item in rows]
    return [{"event": "step", "data": {"run_id": run_id, **step}} for step in list_steps(db, user_id, run_id)]


def _run_response(run_id: int, state: dict[str, Any]) -> dict[str, Any]:
    route_plan = state.get("route_plan") or {}
    return {
        "run_id": run_id,
        "thread_id": state.get("thread_id", ""),
        "status": state.get("status", "completed"),
        "route": state.get("route"),
        "intent": route_plan.get("intent", state.get("route")),
        "route_plan": route_plan.get("route", []),
        "risk_level": route_plan.get("risk_level", "L0"),
        "final_output": state.get("final_output", ""),
        "final_answer": state.get("final_answer", ""),
        "final_payload": state.get("final_payload"),
        "research": state.get("research", {}) or state.get("research_result", {}),
        "rag": state.get("rag", {}) or state.get("rag_result", {}),
        "artifacts": state.get("artifacts", []),
        "memory_updates": state.get("memory_updates", []),
        "skill_drafts": state.get("skill_drafts", []),
        "matched_skill": state.get("matched_skill"),
        "candidate_skills": state.get("candidate_skills", []),
        "created_skill_draft": state.get("created_skill_draft"),
        "reusable_score": (state.get("skill_reuse") or {}).get("reusable_score", 0),
        "tool_call": state.get("tool_call", {}),
        "tool_result": state.get("tool_result"),
        "evaluation": state.get("evaluation", {}),
        "errors": state.get("errors", []),
        "error": state.get("error", ""),
        "approval_required": state.get("approval_required", False),
        "approval_payload": state.get("approval_payload"),
        "agent_outputs": state.get("agent_outputs", []),
    }


def _event_to_sse(item) -> dict[str, Any]:
    return {
        "event": item.event_type,
        "data": {
            "id": item.id,
            "run_id": item.run_id,
            "thread_id": item.thread_id,
            "user_id": item.user_id,
            "event_type": item.event_type,
            "node_name": item.node_name,
            "payload": item.payload_json or {},
            "created_at": item.created_at.isoformat() if item.created_at else "",
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
