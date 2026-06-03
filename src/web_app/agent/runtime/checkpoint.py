from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.agent_repository import AgentStepRepository


def record_step(db: Session, run_id: int, node_name: str, action_type: str, input_data: dict[str, Any], output_data: dict[str, Any], status: str = "completed") -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
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
