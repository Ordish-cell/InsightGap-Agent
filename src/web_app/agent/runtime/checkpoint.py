from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.agent_repository import AgentEventRepository, AgentRunRepository, AgentStepRepository


def record_event(db: Session, run_id: int, event_type: str, payload: dict[str, Any] | None = None, node_name: str = "", user_id: int | None = None, thread_id: str | None = None) -> None:
    run = AgentRunRepository(db).get_by_id(run_id)
    if not run and user_id is None:
        return
    state = (getattr(run, "graph_state", None) or {}) if run else {}
    AgentEventRepository(db).create(
        run_id=run_id,
        thread_id=thread_id or state.get("thread_id", ""),
        user_id=user_id or getattr(run, "user_id", 0),
        event_type=event_type,
        node_name=node_name,
        payload_json=payload or {},
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
