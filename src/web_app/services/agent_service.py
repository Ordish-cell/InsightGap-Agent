import asyncio
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from src.web_app.agent.runtime import AgentRuntime
from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.db.repositories.agent_repository import (
    AgentChatMessageRepository,
    AgentConversationRepository,
    AgentEventRepository,
    AgentRunRepository,
    AgentStepRepository,
)


GENERIC_COMPLETED_ANSWERS = {
    "",
    "agent completed",
    "agent completed, but no displayable output was returned",
    "agent run completed",
    "agent runtime completed",
}


async def run_agent_async(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now()
    user_input = payload.get("user_input") or payload.get("input") or payload.get("query") or ""
    payload = {**payload, "user_input": user_input}
    page_context = payload.get("page_context") or {}
    conversation_repo = AgentConversationRepository(db)
    message_repo = AgentChatMessageRepository(db)
    conversation = _get_or_create_conversation(db, user_id, payload, user_input)
    conversation_id = conversation.conversation_id
    selected_feed_card_id = page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
    selected_feed_card_title = str(page_context.get("selected_feed_card_title") or page_context.get("feed_card_title") or "")
    if selected_feed_card_id and not payload.get("feed_card_id"):
        payload["feed_card_id"] = selected_feed_card_id
    run_repo = AgentRunRepository(db)
    thread_id = conversation.thread_id or f"user:{user_id}:conversation:{conversation_id}"
    if not conversation.thread_id:
        conversation_repo.update(conversation, thread_id=thread_id)
    run = run_repo.create(
        user_id=user_id,
        conversation_id=conversation_id,
        thread_id=thread_id,
        run_type=payload.get("run_type", "agent_runtime"),
        mode=payload.get("mode", "react"),
        status="running",
        user_input=user_input,
        graph_state={"source": payload.get("source", "agent_page"), "page_context": page_context, "thread_id": thread_id, "conversation_id": conversation_id},
    )
    user_message = message_repo.create(
        message_id=str(uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        run_id=run.id,
        thread_id=thread_id,
        role="user",
        content=user_input,
        status="completed",
        metadata_json={"source": payload.get("source", "agent_page"), "page_context": page_context},
    )
    assistant_message = message_repo.create(
        message_id=str(uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        run_id=run.id,
        thread_id=thread_id,
        role="assistant",
        content="",
        status="thinking",
        metadata_json={"source": payload.get("source", "agent_page")},
    )
    record_event(db, run.id, "run_started", {"user_input": user_input, "source": payload.get("source", "agent_page")}, user_id=user_id, thread_id=thread_id)
    try:
        state = await AgentRuntime(db, payload).run({"user_id": user_id, "run_id": run.id, "thread_id": thread_id, "conversation_id": conversation_id, "user_input": user_input, "mode": run.mode, "source": payload.get("source", "agent_page"), "page_context": page_context})
    except Exception as exc:
        state = {
            "user_id": user_id,
            "run_id": run.id,
            "thread_id": thread_id,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "status": "failed",
            "error": str(exc),
            "errors": [str(exc)],
            "final_output": "",
            "langgraphstatus": {
                "run_id": run.id,
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "status": "failed",
                "phase": "failed",
                "summary": "Agent runtime failed before producing a final answer.",
                "steps": [],
            },
        }
    elapsed_ms = max(0, int((datetime.now() - started_at).total_seconds() * 1000))
    answer = build_user_facing_answer(state)
    state["answer"] = answer
    state["final_answer"] = answer
    state["final_output"] = answer
    final_payload = dict(state.get("final_payload") or {})
    final_payload["answer"] = answer
    final_payload.setdefault("run_id", str(run.id))
    final_payload.setdefault("thread_id", thread_id)
    final_payload.setdefault("conversation_id", conversation_id)
    state["final_payload"] = final_payload
    langgraphstatus = dict(state.get("langgraphstatus") or {})
    langgraphstatus.update(
        {
            "run_id": run.id,
            "thread_id": thread_id,
            "conversation_id": conversation_id,
            "status": state.get("status", "completed"),
            "phase": "completed" if state.get("status", "completed") == "completed" else state.get("status", "completed"),
            "elapsed_ms": elapsed_ms,
        }
    )
    state["langgraphstatus"] = langgraphstatus
    steps = list(langgraphstatus.get("steps") or [])
    completed_at = datetime.now() if state.get("status") in {"completed", "failed", "waiting_approval"} else None
    run_repo.update(
        run,
        status=state.get("status", "completed"),
        graph_state=_json_safe(dict(state)),
        result_summary=answer,
        final_answer=answer,
        final_response=_json_safe(final_payload),
        langgraphstatus_json=_json_safe(langgraphstatus),
        elapsed_ms=elapsed_ms,
        error_message=state.get("error", ""),
        completed_at=completed_at,
    )
    message_repo.update(
        assistant_message,
        content=answer,
        status="failed" if state.get("status") == "failed" else "completed" if state.get("status") == "completed" else state.get("status", "completed"),
        elapsed_ms=elapsed_ms,
        langgraphstatus_json=_json_safe(langgraphstatus),
        steps_json=_json_safe(steps),
        error_message=state.get("error", ""),
        metadata_json={"run_id": run.id, "final_response": _json_safe(final_payload)},
    )
    conversation_repo.touch(
        conversation,
        preview=answer,
        last_run_id=run.id,
        selected_feed_card_id=int(selected_feed_card_id) if str(selected_feed_card_id or "").isdigit() else None,
        selected_feed_card_title=selected_feed_card_title or None,
    )
    if state.get("approval_required") or state.get("status") == "waiting_approval":
        record_event(db, run.id, "approval_required", state.get("approval_payload") or {}, user_id=user_id, thread_id=thread_id)
    elif state.get("status") == "failed":
        record_event(db, run.id, "run_failed", {"status": state.get("status"), "answer": answer, "error": state.get("error", "")}, user_id=user_id, thread_id=thread_id)
    else:
        record_event(db, run.id, "final_response_created", {"answer": answer, "answer_len": len(answer)}, user_id=user_id, thread_id=thread_id)
        record_event(db, run.id, "run_completed", {"status": state.get("status"), "answer": answer}, user_id=user_id, thread_id=thread_id)
    return _run_response(run.id, state, conversation=conversation, user_message=user_message, assistant_message=assistant_message, elapsed_ms=elapsed_ms)


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
        "answer": run.final_answer or run.result_summary,
        "final_answer": run.final_answer,
        "final_response": run.final_response or {},
        "langgraphstatus": run.langgraphstatus_json or (run.graph_state or {}).get("langgraphstatus", {}),
        "elapsed_ms": run.elapsed_ms,
        "error_message": run.error_message,
        "graph_state": run.graph_state or {},
        "conversation_id": run.conversation_id or (run.graph_state or {}).get("conversation_id", ""),
        "thread_id": run.thread_id or (run.graph_state or {}).get("thread_id", ""),
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


def create_conversation(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    item = _create_conversation(db, user_id, payload, payload.get("title") or "")
    return _conversation_response(item, messages=[])


def list_conversations(db: Session, user_id: int, status: str = "active", limit: int = 50, offset: int = 0) -> dict[str, Any]:
    items = AgentConversationRepository(db).list_by_user(user_id, status=status, limit=limit, offset=offset)
    return {"items": [_conversation_response(item) for item in items]}


def get_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation = _require_conversation(db, user_id, conversation_id)
    messages = AgentChatMessageRepository(db).list_by_conversation(user_id, conversation_id)
    return _conversation_response(conversation, messages=messages)


def update_conversation(db: Session, user_id: int, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    repo = AgentConversationRepository(db)
    conversation = _require_conversation(db, user_id, conversation_id)
    values: dict[str, Any] = {}
    if "title" in payload:
        values["title"] = str(payload.get("title") or "")[:255]
    if "selected_feed_card_id" in payload:
        values["selected_feed_card_id"] = payload.get("selected_feed_card_id")
    if "selected_feed_card_title" in payload:
        values["selected_feed_card_title"] = str(payload.get("selected_feed_card_title") or "")[:512]
    if "metadata" in payload or "metadata_json" in payload:
        values["metadata_json"] = payload.get("metadata_json") or payload.get("metadata") or {}
    if values:
        conversation = repo.update(conversation, **values)
    return _conversation_response(conversation, messages=AgentChatMessageRepository(db).list_by_conversation(user_id, conversation_id))


def archive_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation = _require_conversation(db, user_id, conversation_id)
    conversation = AgentConversationRepository(db).update(conversation, status="archived")
    return _conversation_response(conversation)


def delete_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation = _require_conversation(db, user_id, conversation_id)
    conversation = AgentConversationRepository(db).update(conversation, status="deleted")
    return _conversation_response(conversation)


def clear_conversation(db: Session, user_id: int, conversation_id: str) -> dict[str, Any]:
    conversation_repo = AgentConversationRepository(db)
    conversation = _require_conversation(db, user_id, conversation_id)
    removed = AgentChatMessageRepository(db).clear_conversation(user_id, conversation_id)
    conversation = conversation_repo.update(conversation, message_count=0, last_message_preview="")
    return {"conversation": _conversation_response(conversation), "cleared_messages": removed}


def build_user_facing_answer(state: dict[str, Any]) -> str:
    final_payload = state.get("final_payload") or {}
    candidates = [
        state.get("answer"),
        final_payload.get("answer") if isinstance(final_payload, dict) else "",
        state.get("final_answer"),
        state.get("final_output"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and not _is_generic_completed_answer(text):
            return text

    status = state.get("status", "completed")
    errors = [str(item) for item in state.get("errors", []) if item] or ([str(state.get("error"))] if state.get("error") else [])
    if status == "failed" or errors:
        reason = errors[0] if errors else "runtime returned a failure state"
        return f"Run failed: {reason}. You can retry or ask me to inspect this Agent Run."

    route_plan = state.get("route_plan") or {}
    home_intent = state.get("home_intent") or {}
    intent = str(route_plan.get("intent") or home_intent.get("intent") or state.get("route") or "chat")
    risk_level = str(route_plan.get("risk_level") or home_intent.get("risk_level") or "L0")
    if status == "waiting_approval" or route_plan.get("needs_approval") or home_intent.get("needs_approval"):
        return f"Approval required: this is a {risk_level} risk action and must be approved before execution. I have not performed any external write or irreversible operation."

    user_input = str(state.get("user_input") or "").strip()
    if _looks_like_greeting(user_input):
        return "\u4f60\u597d\uff0c\u6211\u662f\u4fe1\u606f\u5dee Agent OS \u52a9\u624b\u3002\u4f60\u53ef\u4ee5\u8ba9\u6211\u7814\u7a76\u4fe1\u606f\u3001\u751f\u6210\u6210\u679c\u6216\u6c89\u6dc0 Skill\u3002"
    if intent == "research":
        return "\u6211\u5df2\u8bc6\u522b\u8fd9\u662f\u7814\u7a76\u4efb\u52a1\uff0c\u5e76\u5b8c\u6210\u4e86\u521d\u6b65\u89c4\u5212\u3002\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u5b8c\u6574\u7814\u7a76\u7ed3\u679c\u3002"
    if intent == "artifact":
        return "\u6211\u5df2\u8bc6\u522b\u8fd9\u662f\u6210\u679c\u751f\u6210\u4efb\u52a1\uff0c\u4f46\u5f53\u524d\u8fd8\u6ca1\u6709\u751f\u6210\u5b9e\u9645 Artifact\u3002"
    if intent == "tool":
        return "\u6211\u5df2\u8bc6\u522b\u8fd9\u662f\u5de5\u5177\u76f8\u5173\u4efb\u52a1\u3002\u5982\u679c\u6d89\u53ca\u5916\u90e8\u5199\u5165\u6216\u9ad8\u98ce\u9669\u52a8\u4f5c\uff0c\u4f1a\u5148\u8fdb\u5165\u5ba1\u6279\u72b6\u6001\u3002"

    return "\u6211\u5df2\u7ecf\u5b8c\u6210\u57fa\u7840\u5224\u65ad\u3002\u4f60\u53ef\u4ee5\u7ee7\u7eed\u8865\u5145\u76ee\u6807\uff0c\u6211\u4f1a\u6cbf\u7528\u5f53\u524d\u4f1a\u8bdd\u4e0a\u4e0b\u6587\u3002"

def _run_response(
    run_id: int,
    state: dict[str, Any],
    *,
    conversation=None,
    user_message=None,
    assistant_message=None,
    elapsed_ms: int = 0,
) -> dict[str, Any]:
    route_plan = state.get("route_plan") or {}
    answer = build_user_facing_answer(state)
    final_response = dict(state.get("final_payload") or {})
    final_response["answer"] = answer
    return {
        "run_id": run_id,
        "conversation_id": state.get("conversation_id", ""),
        "thread_id": state.get("thread_id", ""),
        "status": state.get("status", "completed"),
        "elapsed_ms": elapsed_ms,
        "answer": answer,
        "route": state.get("route"),
        "intent": route_plan.get("intent", state.get("route")),
        "route_plan": route_plan.get("route", []),
        "risk_level": route_plan.get("risk_level", "L0"),
        "final_output": answer,
        "final_answer": answer,
        "final_response": final_response,
        "final_payload": final_response,
        "langgraphstatus": state.get("langgraphstatus", {}),
        "user_message": _message_response(user_message) if user_message else None,
        "assistant_message": _message_response(assistant_message) if assistant_message else None,
        "conversation": _conversation_response(conversation) if conversation else None,
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


def _get_or_create_conversation(db: Session, user_id: int, payload: dict[str, Any], user_input: str):
    conversation_id = str(payload.get("conversation_id") or "").strip()
    if conversation_id:
        return _require_conversation(db, user_id, conversation_id)
    return _create_conversation(db, user_id, payload, user_input)


def _create_conversation(db: Session, user_id: int, payload: dict[str, Any], title_seed: str):
    conversation_id = str(uuid4())
    thread_id = f"user:{user_id}:conversation:{conversation_id}"
    page_context = payload.get("page_context") or {}
    selected_feed_card_id = page_context.get("selected_feed_card_id") or page_context.get("feed_card_id")
    try:
        selected_feed_card_id = int(selected_feed_card_id) if selected_feed_card_id else None
    except (TypeError, ValueError):
        selected_feed_card_id = None
    selected_feed_card_title = str(page_context.get("selected_feed_card_title") or page_context.get("feed_card_title") or "")
    title = str(payload.get("title") or title_seed or "New Agent Run").strip()
    return AgentConversationRepository(db).create(
        conversation_id=conversation_id,
        user_id=user_id,
        title=_short_title(title),
        source=payload.get("source", "agent_page"),
        status="active",
        thread_id=thread_id,
        selected_feed_card_id=selected_feed_card_id,
        selected_feed_card_title=selected_feed_card_title[:512],
        metadata_json={"page_context": page_context},
        last_message_preview="",
        message_count=0,
    )


def _require_conversation(db: Session, user_id: int, conversation_id: str):
    conversation = AgentConversationRepository(db).get_by_conversation_id(user_id, conversation_id)
    if not conversation:
        raise ValueError("Agent conversation not found")
    return conversation


def _short_title(value: str) -> str:
    title = " ".join(value.split())
    if not title:
        return "New Agent Run"
    return title[:42]


def _looks_like_greeting(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"hi", "hello", "hey"} or any(token in value for token in ("\u4f60\u597d", "\u60a8\u597d", "\u4f60\u662f\u8c01", "\u4f60\u662f\u8ab0"))


def _is_generic_completed_answer(value: str) -> bool:
    normalized = value.strip().rstrip(".\u3002").lower()
    if normalized in GENERIC_COMPLETED_ANSWERS:
        return True
    return normalized.startswith("\u5df2\u5b8c\u6210\u672c\u6b21 agent run") or normalized.startswith("\u5df2\u5b8c\u6210\u672c\u6b21 agent")

def _conversation_response(item, messages: list[Any] | None = None) -> dict[str, Any]:
    if not item:
        return {}
    data = {
        "id": item.id,
        "conversation_id": item.conversation_id,
        "thread_id": item.thread_id,
        "title": item.title,
        "source": item.source,
        "status": item.status,
        "selected_feed_card_id": item.selected_feed_card_id,
        "selected_feed_card_title": item.selected_feed_card_title,
        "metadata": item.metadata_json or {},
        "last_message_preview": item.last_message_preview,
        "last_run_id": item.last_run_id,
        "message_count": item.message_count,
        "created_at": item.created_at.isoformat() if item.created_at else "",
        "updated_at": item.updated_at.isoformat() if item.updated_at else "",
        "last_active_at": item.last_active_at.isoformat() if item.last_active_at else "",
    }
    if messages is not None:
        data["messages"] = [_message_response(message) for message in messages]
    return data


def _message_response(item) -> dict[str, Any]:
    if not item:
        return {}
    return {
        "id": item.id,
        "message_id": item.message_id,
        "conversation_id": item.conversation_id,
        "run_id": item.run_id,
        "thread_id": item.thread_id,
        "role": item.role,
        "content": item.content,
        "status": item.status,
        "elapsed_ms": item.elapsed_ms,
        "langgraphstatus": item.langgraphstatus_json or {},
        "steps": item.steps_json or [],
        "error_message": item.error_message,
        "metadata": item.metadata_json or {},
        "created_at": item.created_at.isoformat() if item.created_at else "",
        "updated_at": item.updated_at.isoformat() if item.updated_at else "",
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
