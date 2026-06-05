from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import AgentEvent, AgentRun, AgentStep, LLMCall


class AgentRunRepository(BaseRepository[AgentRun]):
    model = AgentRun

    def get_by_user(self, user_id: int, run_id: int) -> AgentRun | None:
        return self.db.execute(select(AgentRun).where(AgentRun.user_id == user_id, AgentRun.id == run_id)).scalar_one_or_none()


class AgentStepRepository(BaseRepository[AgentStep]):
    model = AgentStep

    def list_by_run(self, run_id: int) -> list[AgentStep]:
        return list(self.db.execute(select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.id)).scalars())


class AgentEventRepository(BaseRepository[AgentEvent]):
    model = AgentEvent

    def list_by_run(self, user_id: int, run_id: int) -> list[AgentEvent]:
        stmt = select(AgentEvent).where(AgentEvent.user_id == user_id, AgentEvent.run_id == run_id).order_by(AgentEvent.id)
        return list(self.db.execute(stmt).scalars())


class LLMCallRepository(BaseRepository[LLMCall]):
    model = LLMCall

    def list_by_run(self, user_id: int, run_id: int) -> list[LLMCall]:
        stmt = select(LLMCall).where(LLMCall.user_id == user_id, LLMCall.run_id == run_id).order_by(LLMCall.id)
        return list(self.db.execute(stmt).scalars())
