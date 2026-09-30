"""Durable opt-in memory extraction; no response-path model calls."""
import asyncio
import logging
from contextvars import copy_context

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.web_app.db.session import SessionLocal
from src.web_app.models.orm import AgentRun, AgentChatMessage, AgentConversation, MemoryMaintenanceTask, Memory
from src.web_app.memory.policy import settings_for

logger = logging.getLogger(__name__)


def retry_indexes():
    """Repair committed facts after index outages, including interrupted indexing."""
    from src.web_app.core.config import settings
    from src.web_app.services.memory_service import memory_service
    from src.web_app.services.deletion_guard import conversation_write, ConversationDeletingError
    vector = bool(settings.qdrant_url)
    graph = bool(settings.enable_neo4j and settings.neo4j_memory_graph_enabled)
    if not vector and not graph:
        return
    with SessionLocal() as db:
        from sqlalchemy import or_
        markers = [Memory.metadata_json["index_pending"].as_boolean().is_(True)]
        if vector:
            markers.append(Memory.metadata_json["qdrant_indexed"].as_boolean().is_(None))
        if graph:
            markers.append(Memory.metadata_json["graph_indexed"].as_boolean().is_(None))
        rows = list(db.scalars(select(Memory).where(or_(*markers)).order_by(Memory.updated_at).limit(200)))
        for item in rows:
            if not memory_service.is_recallable(item, db, item.scope_id):
                item.metadata_json = {**(item.metadata_json or {}), "index_pending": False,
                                      "qdrant_indexed": False, "graph_indexed": False}
                db.commit()
                continue
            try:
                with conversation_write(db, item.user_id, item.scope_id if item.scope == "conversation" else None,
                                        require_exists=item.scope == "conversation"):
                    memory_service._index_saved(item, db)
            except ConversationDeletingError:
                continue


def still_allowed(db, job):
    db.expire_all()
    conversation = db.scalar(select(AgentConversation).where(
        AgentConversation.user_id == job.user_id, AgentConversation.conversation_id == job.conversation_id))
    if not conversation or conversation.status in {"deleting", "deleted"}:
        return False
    current = settings_for(db, job.user_id, job.conversation_id, job.policy.get("request_overrides", {}))
    return bool(current["generate_memory"] and not job.policy.get("writes_blocked")
                and current["profile_version"] == job.policy.get("profile_version")
                and current["conversation_version"] == job.policy.get("conversation_version"))


def process(job_id):
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.services.llm_registry_service import resolve_model_context
    from src.web_app.memory.extractor import LlmMemoryExtractor
    from src.web_app.memory.tokens import chunks
    from src.web_app.services.memory_service import memory_service
    from src.web_app.services.deletion_guard import conversation_write
    from src.web_app.agent.runtime.chat_control import execution
    execution.set(None)
    with SessionLocal() as db:
        job = db.get(MemoryMaintenanceTask, job_id)
        if not job or job.status in {"completed", "cancelled"} or job.attempts >= 3:
            return
        if not still_allowed(db, job):
            job.status = "cancelled"
            db.commit()
            return
        job.status, job.attempts = "running", job.attempts + 1
        db.commit()
        try:
            run = db.get(AgentRun, job.run_id)
            source = db.get(AgentChatMessage, job.source_message_id)
            if (not run or run.status != "completed" or run.user_id != job.user_id or run.conversation_id != job.conversation_id
                or not source or source.status != "completed" or source.role != "user" or source.user_id != job.user_id
                or source.conversation_id != job.conversation_id or source.run_id != run.id):
                job.status = "cancelled"
                db.commit()
                return
            context = resolve_model_context(db, job.user_id, (run.graph_state or {}).get("model_config_id"), job.conversation_id)
            extraction = {"semantic_memories": [], "episodic_memories": [], "working_memories": []}
            with use_model_context(context):
                for part in chunks(source.content):
                    result = asyncio.run(LlmMemoryExtractor().extract(db=db, run_id=str(run.id),
                        user_id=job.user_id, user_input=part, agent_output="", strict=True))
                    for kind in ("semantic_memories", "episodic_memories"):
                        extraction[kind].extend(result.get(kind, []))
                # Final settings/deletion check and writes share deletion's serialization boundary.
                with conversation_write(db, job.user_id, job.conversation_id, require_exists=True):
                    if not still_allowed(db, job):
                        job.status = "cancelled"
                        db.commit()
                        return
                    saved = memory_service._save_extracted(job.user_id, extraction, db, str(run.id))
                    summary = memory_service.consolidate_memory(job.user_id, db, conversation_id=job.conversation_id)
                    job.status, job.error_message = "completed", ""
                    job.result = {"total_saved": saved["total_saved"], "summary_count": summary["summary_count"]}
                    db.commit()
        except Exception as exc:
            db.rollback()
            job = db.get(MemoryMaintenanceTask, job_id)
            if job:
                job.status, job.error_message = "failed", type(exc).__name__
                db.commit()
            logger.exception("memory.maintenance_failed job_id=%s", job_id)


class MemoryTasks:
    def __init__(self):
        self.tasks = {}
        self.slots = asyncio.Semaphore(2)
        self.closing = False
        self.index_task = None

    def enqueue(self, db, user_id, conversation_id, run_id):
        run = db.get(AgentRun, run_id)
        policy = (run.memory_policy or {}) if run else {}
        if not run or run.status != "completed" or not policy.get("generate_memory") or policy.get("writes_blocked"):
            return
        source = db.scalar(select(AgentChatMessage).where(AgentChatMessage.user_id == user_id,
            AgentChatMessage.conversation_id == conversation_id, AgentChatMessage.run_id == run_id,
            AgentChatMessage.role == "user", AgentChatMessage.status == "completed").order_by(AgentChatMessage.id).limit(1))
        if not source:
            return
        existing = db.scalar(select(MemoryMaintenanceTask).where(MemoryMaintenanceTask.run_id == run_id))
        if not existing:
            job = MemoryMaintenanceTask(user_id=user_id, conversation_id=conversation_id, run_id=run_id,
                source_message_id=source.id, policy=dict(policy), status="pending", attempts=0)
            db.add(job)
            try:
                db.commit()
            except SQLAlchemyError:
                db.rollback()
                logger.exception("memory.enqueue_failed run_id=%s", run_id)
                return  # the completed run retains consent for recovery
        self.start(user_id, conversation_id)

    def start(self, user_id, conversation_id):
        key = (user_id, conversation_id)
        if self.closing or key in self.tasks:
            return
        context = copy_context()
        from src.web_app.agent.runtime.chat_control import execution
        context.run(execution.set, None)
        self.tasks[key] = asyncio.create_task(self._drain(key), context=context)

    async def _drain(self, key):
        try:
            while not self.closing:
                with SessionLocal() as db:
                    job_id = db.scalar(select(MemoryMaintenanceTask.id).where(
                        MemoryMaintenanceTask.user_id == key[0], MemoryMaintenanceTask.conversation_id == key[1],
                        MemoryMaintenanceTask.status.in_(["pending", "running", "failed"]),
                        MemoryMaintenanceTask.attempts < 3).order_by(MemoryMaintenanceTask.source_message_id).limit(1))
                if job_id is None:
                    return
                async with self.slots:
                    await asyncio.to_thread(process, job_id)
                await asyncio.sleep(1)
        finally:
            self.tasks.pop(key, None)
            # Enqueue can arrive after the last query but before cleanup.
            if not self.closing:
                with SessionLocal() as db:
                    pending = db.scalar(select(MemoryMaintenanceTask.id).where(
                        MemoryMaintenanceTask.user_id == key[0], MemoryMaintenanceTask.conversation_id == key[1],
                        MemoryMaintenanceTask.status == "pending").limit(1))
                if pending:
                    self.start(*key)

    def recover(self):
        if self.index_task is None or self.index_task.done():
            self.index_task = asyncio.create_task(self._repair_indexes())
        self._recover_missing()
        with SessionLocal() as db:
            for job in db.scalars(select(MemoryMaintenanceTask).where(MemoryMaintenanceTask.status == "running")):
                job.status, job.error_message = "failed", "worker_interrupted"
            db.commit()
            keys = db.execute(select(MemoryMaintenanceTask.user_id, MemoryMaintenanceTask.conversation_id).where(
                MemoryMaintenanceTask.status.in_(["pending", "running", "failed"]), MemoryMaintenanceTask.attempts < 3).distinct()).all()
        for key in keys:
            self.start(*key)

    def _recover_missing(self):
        # Close the crash window between a committed reply and job creation.
        # Legacy runs have no recorded opt-in policy and are never re-extracted.
        with SessionLocal() as db:
            runs = list(db.scalars(select(AgentRun).outerjoin(MemoryMaintenanceTask,
                MemoryMaintenanceTask.run_id == AgentRun.id).where(MemoryMaintenanceTask.id.is_(None),
                AgentRun.status == "completed", AgentRun.memory_policy["generate_memory"].as_boolean().is_(True))
                .order_by(AgentRun.id)))
            for run in runs:
                self.enqueue(db, run.user_id, run.conversation_id, run.id)

    async def _repair_indexes(self):
        while not self.closing:
            try:
                self._recover_missing()
                async with self.slots:
                    await asyncio.to_thread(retry_indexes)
            except Exception:
                logger.exception("memory.index_recovery_failed")
            await asyncio.sleep(30)

    async def shutdown(self):
        self.closing = True
        if self.index_task:
            self.index_task.cancel()
            await asyncio.gather(self.index_task, return_exceptions=True)
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks.values()))


memory_tasks = MemoryTasks()
