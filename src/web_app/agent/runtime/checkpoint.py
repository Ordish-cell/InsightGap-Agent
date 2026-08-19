from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.events import event_channels, validate_event_payload
from src.web_app.db.repositories.agent_repository import AgentEventRepository, AgentRunRepository, AgentStepRepository
from src.web_app.models.orm import AgentEvent


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
    run = AgentRunRepository(db).get_by_id(run_id)
    if not run and user_id is None:
        return None
    state = (getattr(run, "graph_state", None) or {}) if run else {}
    resolved_visibility, resolved_channel = event_channels(event_type)
    validated_payload = validate_event_payload(event_type, payload or {})
    return AgentEventRepository(db).create(
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


def record_step(db: Session, run_id: int, node_name: str, action_type: str, input_data: dict[str, Any], output_data: dict[str, Any], status: str = "completed") -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    record_event(db, run_id, "node_started", {"action_type": action_type, "input": input_data}, node_name=node_name)
    AgentStepRepository(db).create(
        run_id=run_id,
        node_name=node_name,
        agent_name="langgraph_runtime",
        action_type=action_type,
        input=input_data,
        output=output_data,
        status=status,
        started_at=now,
        ended_at=now,
    )
    event_type = "node_failed" if status == "failed" else "node_completed"
    record_event(db, run_id, event_type, {"action_type": action_type, "status": status, "output": output_data}, node_name=node_name)
