from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.events import event_channels, validate_event_payload
from src.web_app.agent.runtime.chat_control import serialized_transition
from src.web_app.db.repositories.agent_repository import AgentEventRepository, AgentRunRepository, AgentStepRepository
from src.web_app.models.orm import AgentEvent


@serialized_transition
def record_event(
    db: Session,
    run_id: int,
    event_type: str,
    payload: dict[str, Any] | None = None,
    node_name: str = "",
    user_id: int | None = None,
    thread_id: str | None = None,
    *,
    schema_version: int = 1,
    visibility: str | None = None,
    display_channel: str | None = None,
) -> AgentEvent | None:
    from src.web_app.agent.runtime.chat_control import check_active
    if event_type != "node_cancelled":
        check_active(run_id)
    run = AgentRunRepository(db).get_by_id(run_id)
    if not run and user_id is None:
        return None
    state = (getattr(run, "graph_state", None) or {}) if run else {}
    resolved_visibility, resolved_channel = event_channels(event_type)
    validated_payload = validate_event_payload(event_type, payload or {})
    if event_type in {"answer_started", "answer_delta", "answer_completed"} and run:
        from sqlalchemy import select
        from src.web_app.models.orm import AgentChatMessage
        message_id = db.execute(select(AgentChatMessage.message_id).where(
            AgentChatMessage.run_id == run_id, AgentChatMessage.role == "assistant",
        )).scalar_one_or_none()
        if message_id:
            validated_payload["message_id"] = message_id
    from src.web_app.agent.runtime.live_progress import current_step
    if event_type.startswith("node_") and current_step.get() and not validated_payload.get("step_id"):
        return None  # Legacy node-local lifecycle duplicates the execution wrapper.
    event = AgentEventRepository(db).create(
        run_id=run_id,
        thread_id=thread_id or state.get("thread_id", ""),
        user_id=user_id or getattr(run, "user_id", 0),
        event_type=event_type,
        node_name=node_name,
        schema_version=schema_version,
        visibility=visibility or resolved_visibility,
        display_channel=display_channel or resolved_channel,
        payload_json=validated_payload,
    )
    from src.web_app.agent.runtime.event_notify import notify
    notify(run_id)
    return event


@serialized_transition
def record_step(db: Session, run_id: int, node_name: str, action_type: str, input_data: dict[str, Any], output_data: dict[str, Any], status: str = "completed") -> None:
    from src.web_app.agent.runtime.live_progress import step_started_at
    now = datetime.now(UTC).replace(tzinfo=None)
    AgentStepRepository(db).create(
        run_id=run_id,
        node_name=node_name,
        agent_name="langgraph_runtime",
        action_type=action_type,
        input=input_data,
        output=output_data,
        status=status,
        started_at=step_started_at.get() or now,
        ended_at=now,
    )
    record_event(db, run_id, "step_recorded", {"action_type": action_type, "status": status, "output": output_data}, node_name=node_name)
