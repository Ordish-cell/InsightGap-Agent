from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update

from sqlalchemy import delete as sa_delete

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import AgentChatMessage, AgentConversation, AgentEvent, AgentRun, AgentStep, Approval, Artifact, Document, DocumentChunk, LLMCall, Memory, ResearchRun, ToolCall


class AgentRunRepository(BaseRepository[AgentRun]):
    model = AgentRun

    def get_by_user(self, user_id: int, run_id: int) -> AgentRun | None:
        return self.db.execute(select(AgentRun).where(AgentRun.user_id == user_id, AgentRun.id == run_id)).scalar_one_or_none()

    def list_by_conversation(self, user_id: int, conversation_id: str) -> list[AgentRun]:
        stmt = select(AgentRun).where(AgentRun.user_id == user_id, AgentRun.conversation_id == conversation_id)
        return list(self.db.execute(stmt).scalars())

    def hard_delete_cascade(self, run: AgentRun) -> None:
        """Delete a run and all its child records (steps, events, llm_calls, tool_calls)."""
        run_id = run.id
        # Use core-level bulk deletes for children — they execute immediately
        # so FK constraints are resolved before the parent is removed.
        self.db.execute(delete(AgentStep).where(AgentStep.run_id == run_id))
        self.db.execute(delete(AgentEvent).where(AgentEvent.run_id == run_id))
        self.db.execute(delete(LLMCall).where(LLMCall.run_id == run_id))
        self.db.execute(delete(ToolCall).where(ToolCall.run_id == run_id))
        self.db.delete(run)
        self.db.commit()


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

    def hard_delete(self, user_id: int, conversation_id: str) -> int:
        """Hard-delete a conversation and all associated records.

        Cascade includes: messages, runs, steps, events, llm_calls, tool_calls,
        documents (and chunks), and working conversations memories.
        Returns total count of deleted records.
        """
        deleted = 0
        conversation = self.db.execute(
            select(AgentConversation).where(
                AgentConversation.user_id == user_id,
                AgentConversation.conversation_id == conversation_id,
            )
        ).scalar_one_or_none()
        if not conversation:
            return 0

        run_repo = AgentRunRepository(self.db)
        msg_repo = AgentChatMessageRepository(self.db)
        runs = run_repo.list_by_conversation(user_id, conversation_id)
        run_ids = [run.id for run in runs]

        # ── 0) Collect document IDs from chat attachments ──────────────
        doc_ids: set[int] = set()
        messages = msg_repo.list_by_conversation(user_id, conversation_id)
        for msg in messages:
            meta = msg.metadata_json or {}
            attachments = meta.get("attachments") or []
            for att in attachments:
                did = att.get("document_id") if isinstance(att, dict) else None
                if did:
                    try:
                        doc_ids.add(int(did))
                    except (TypeError, ValueError):
                        pass

        # 1) Delete chat messages first — they have FK run_id → agent_runs
        deleted += msg_repo.hard_delete_by_conversation(user_id, conversation_id)

        # 2) NULL out FK references from tables we're NOT deleting
        if run_ids:
            self.db.execute(update(Approval).where(Approval.run_id.in_(run_ids)).values(run_id=None))
            self.db.execute(update(Artifact).where(Artifact.run_id.in_(run_ids)).values(run_id=None))
            self.db.execute(update(ResearchRun).where(ResearchRun.agent_run_id.in_(run_ids)).values(agent_run_id=None))
            self.db.execute(
                update(AgentConversation)
                .where(AgentConversation.id == conversation.id, AgentConversation.last_run_id.in_(run_ids))
                .values(last_run_id=None)
            )

        # 3) Cascade-delete each run (steps, events, llm_calls, tool_calls → run)
        for run_id in run_ids:
            run = run_repo.get_by_id(run_id)
            if run:
                run_repo.hard_delete_cascade(run)
                deleted += 1

        # 4) Delete documents and chunks for this conversation
        if doc_ids:
            doc_list = list(doc_ids)
            self.db.execute(
                sa_delete(DocumentChunk).where(
                    DocumentChunk.user_id == user_id,
                    DocumentChunk.document_id.in_(doc_list),
                )
            )
            self.db.execute(
                sa_delete(Document).where(
                    Document.user_id == user_id,
                    Document.id.in_(doc_list),
                )
            )
            deleted += len(doc_ids)

        # 5) Delete working / conversation-scoped episodic memories
        mem_stmt = sa_delete(Memory).where(
            Memory.user_id == user_id,
            Memory.memory_type.in_(["working"]),
        )
        result = self.db.execute(mem_stmt)
        deleted += result.rowcount or 0

        # 6) Delete the conversation itself
        self.db.delete(conversation)
        self.db.commit()
        deleted += 1
        return deleted

    def get_conversation_document_ids(self, user_id: int, conversation_id: str) -> set[int]:
        """Return document IDs from chat attachment metadata."""
        doc_ids: set[int] = set()
        messages = AgentChatMessageRepository(self.db).list_by_conversation(user_id, conversation_id)
        for msg in messages:
            meta = msg.metadata_json or {}
            attachments = meta.get("attachments") or []
            for att in attachments:
                did = att.get("document_id") if isinstance(att, dict) else None
                if did:
                    try:
                        doc_ids.add(int(did))
                    except (TypeError, ValueError):
                        pass
        return doc_ids


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

    def list_recent_by_conversation(self, user_id: int, conversation_id: str, limit: int = 12) -> list[AgentChatMessage]:
        """Return the most recent messages for a conversation in chronological order.

        Used by context_builder to inject [Conversation History] so the LLM can
        answer "what did I just ask?" / "did I mention X?" questions from the
        same conversation.
        """
        stmt = (
            select(AgentChatMessage)
            .where(
                AgentChatMessage.user_id == user_id,
                AgentChatMessage.conversation_id == conversation_id,
            )
            .order_by(AgentChatMessage.created_at.desc(), AgentChatMessage.id.desc())
            .limit(limit)
        )
        rows = list(self.db.execute(stmt).scalars())
        return list(reversed(rows))

    def clear_conversation(self, user_id: int, conversation_id: str) -> int:
        rows = self.list_by_conversation(user_id, conversation_id)
        count = len(rows)
        for row in rows:
            self.db.delete(row)
        self.db.commit()
        return count

    def hard_delete_by_conversation(self, user_id: int, conversation_id: str) -> int:
        rows = self.list_by_conversation(user_id, conversation_id)
        count = len(rows)
        if count:
            self.db.execute(
                delete(AgentChatMessage).where(
                    AgentChatMessage.user_id == user_id,
                    AgentChatMessage.conversation_id == conversation_id,
                )
            )
            self.db.commit()
        return count
