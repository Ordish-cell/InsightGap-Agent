"""Bounded, coalescing memory maintenance outside the response critical path."""
import asyncio
import logging
from contextvars import copy_context
from time import perf_counter

from src.web_app.agent.runtime.chat_control import execution
from src.web_app.core.config import settings
from src.web_app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def update_pending(user_id, conversation_id):
    from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository
    from src.web_app.services.conversation_summary_service import conversation_summary_service as service
    started = perf_counter()
    with SessionLocal() as db:
        from src.web_app.services.deletion_guard import check_conversation, ConversationDeletingError
        try:
            check_conversation(db, user_id, conversation_id, require_exists=True)
        except ConversationDeletingError:
            return
        summary = service.get_summary(conversation_id, user_id, db=db) or {}
        cursor = summary.get("last_message_id") or 0
        messages = AgentChatMessageRepository(db).list_by_conversation(user_id, conversation_id)
        completed_runs = {m.run_id for m in messages if m.role == "assistant" and m.status == "completed"}
        pending = [m for m in messages if m.id > cursor and m.run_id in completed_runs
                   and m.status == "completed" and m.role in {"user", "assistant"} and m.content.strip()]
        if pending:
            service.update_after_turn(conversation_id, user_id,
                [{"role": m.role, "content": m.content} for m in pending], db=db,
                last_message_id=max(m.id for m in pending))
        service.create_segment_if_needed(conversation_id=conversation_id, user_id=user_id, db=db)
    logger.info("summary_background conversation_id=%s elapsed_ms=%.1f", conversation_id, (perf_counter() - started) * 1000)


class SummaryTasks:
    def __init__(self):
        self.tasks = {}
        self.pending = {}
        self.slots = asyncio.Semaphore(2)
        self.closing = False

    def schedule(self, user_id, conversation_id):
        if self.closing or not conversation_id or not settings.enable_conversation_summary:
            return
        key = (user_id, conversation_id)
        context = copy_context()
        context.run(execution.set, None)
        self.pending[key] = context
        if key not in self.tasks:
            self.tasks[key] = asyncio.create_task(self._drain(key), context=context)

    async def _drain(self, key):
        try:
            while key in self.pending:
                async with self.slots:
                    if key not in self.pending:
                        return
                    context = self.pending.pop(key).copy()
                    try:
                        # Keep the slot/ordering until the thread actually exits.
                        # The provider's configured request timeout bounds calls.
                        await asyncio.to_thread(context.run, update_pending, *key)
                    except Exception:
                        logger.exception("summary_background_failed conversation_id=%s", key[1])
        finally:
            self.tasks.pop(key, None)

    async def shutdown(self):
        self.closing = True
        self.pending.clear()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks.values()), return_exceptions=True)


summary_tasks = SummaryTasks()
