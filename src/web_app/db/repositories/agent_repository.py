from datetime import UTC, datetime

from sqlalchemy import func, select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import AgentChatMessage, AgentConversation, AgentEvent, AgentRun, AgentStep, LLMCall


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


class AgentConversationRepository(BaseRepository[AgentConversation]):
    model = AgentConversation

    def get_by_conversation_id(self, user_id: int, conversation_id: str) -> AgentConversation | None:
        stmt = select(AgentConversation).where(
            AgentConversation.user_id == user_id,
            AgentConversation.conversation_id == conversation_id,
            AgentConversation.status != "deleted",
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_by_user(self, user_id: int, status: str = "active", limit: int = 50, offset: int = 0) -> list[AgentConversation]:
        stmt = select(AgentConversation).where(AgentConversation.user_id == user_id)
        if status:
            stmt = stmt.where(AgentConversation.status == status)
        else:
            stmt = stmt.where(AgentConversation.status != "deleted")
        stmt = stmt.order_by(AgentConversation.last_active_at.desc(), AgentConversation.id.desc()).offset(offset).limit(limit)
        return list(self.db.execute(stmt).scalars())

    def touch(
        self,
        conversation: AgentConversation,
        *,
        preview: str | None = None,
        last_run_id: int | None = None,
        selected_feed_card_id: int | None = None,
        selected_feed_card_title: str | None = None,
    ) -> AgentConversation:
        values = {"last_active_at": datetime.now(UTC).replace(tzinfo=None)}
        if preview is not None:
            values["last_message_preview"] = preview[:400]
        if last_run_id is not None:
            values["last_run_id"] = last_run_id
        if selected_feed_card_id is not None:
            values["selected_feed_card_id"] = selected_feed_card_id
        if selected_feed_card_title is not None:
            values["selected_feed_card_title"] = selected_feed_card_title[:512]
        values["message_count"] = self.count_messages(conversation.user_id, conversation.conversation_id)
        return self.update(conversation, **values)

    def count_messages(self, user_id: int, conversation_id: str) -> int:
        stmt = select(func.count(AgentChatMessage.id)).where(
            AgentChatMessage.user_id == user_id,
            AgentChatMessage.conversation_id == conversation_id,
        )
        return int(self.db.execute(stmt).scalar() or 0)


class AgentChatMessageRepository(BaseRepository[AgentChatMessage]):
    model = AgentChatMessage

    def get_by_message_id(self, user_id: int, message_id: str) -> AgentChatMessage | None:
        stmt = select(AgentChatMessage).where(AgentChatMessage.user_id == user_id, AgentChatMessage.message_id == message_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_by_conversation(self, user_id: int, conversation_id: str) -> list[AgentChatMessage]:
        stmt = (
            select(AgentChatMessage)
            .where(AgentChatMessage.user_id == user_id, AgentChatMessage.conversation_id == conversation_id)
            .order_by(AgentChatMessage.created_at.asc(), AgentChatMessage.id.asc())
        )
        return list(self.db.execute(stmt).scalars())

    def clear_conversation(self, user_id: int, conversation_id: str) -> int:
        rows = self.list_by_conversation(user_id, conversation_id)
        count = len(rows)
        for row in rows:
            self.db.delete(row)
        self.db.commit()
        return count
