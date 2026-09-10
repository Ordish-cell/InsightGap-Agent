"""Durable, scoped deletion. External work is idempotent; SQL removal is atomic."""
import asyncio
from sqlalchemy import select, delete, update, inspect
from src.web_app.models.orm import (ConversationDeletionTask as Job, AgentConversation, AgentChatMessage,
    AgentConversationSummary, AgentConversationSummarySegment, AgentRun, AgentRunControl, AgentStep,
    AgentEvent, LLMCall, ToolCall, Approval, Artifact, EvalRecord, ResearchRun, Memory, Document, DocumentChunk, Skill)
from src.web_app.db.session import SessionLocal
from src.web_app.services.deletion_guard import asset_lock

TERMINAL = {"completed", "completed_with_warnings"}


class DeletionError(Exception):
    def __init__(self, code, message, status=409):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def require_schema(db):
    if not inspect(db.get_bind()).has_table(Job.__tablename__):
        raise DeletionError("DELETION_MIGRATION_REQUIRED", "请先应用 20260909_0015 数据库迁移，再执行彻底删除。", 503)


def public(job):
    return {"id": job.id, "conversation_id": job.conversation_id, "status": job.status, "phase": job.phase,
            "attempts": job.attempts, "result": job.result, "error_message": job.error_message}


def request_deletion(db, user_id, conversation_id, cancel_pending=False):
    require_schema(db)
    from src.web_app.agent.runtime.chat_control import transition_lock
    with transition_lock, asset_lock:
        existing = db.scalar(select(Job).where(Job.user_id == user_id, Job.conversation_id == conversation_id))
        if existing:
            return public(existing)
        conversation = db.scalar(select(AgentConversation).where(AgentConversation.user_id == user_id,
            AgentConversation.conversation_id == conversation_id).with_for_update())
        if not conversation:
            raise DeletionError("CONVERSATION_NOT_FOUND", "会话不存在。", 404)
        runs = list(db.scalars(select(AgentRun).where(AgentRun.user_id == user_id, AgentRun.conversation_id == conversation_id)))
        run_ids = [r.id for r in runs]
        approvals = list(db.scalars(select(Approval).where(Approval.run_id.in_(run_ids), Approval.status == "pending")))
        if approvals and not cancel_pending:
            raise DeletionError("CONVERSATION_HAS_PENDING_APPROVAL", "当前会话有待审批操作，请先取消审批后删除。")
        from src.web_app.services.agent_run_task_manager import agent_run_task_manager
        for run in runs:
            if run.status in {"created", "running", "resuming"} or agent_run_task_manager.is_running(run.id):
                if run.chat_control_phase != "enabled":
                    raise DeletionError("CONVERSATION_BUSY", "研究或工具仍在执行，结束后才能删除。")
        for approval in approvals:
            approval.status = "cancelled"
        for run in runs:
            if run.status in {"waiting_approval", "paused"}:
                run.status = "cancelled"
        job = Job(user_id=user_id, conversation_id=conversation_id, status="pending", phase="stopping")
        db.add(job)
        conversation.status = "deleting"
        db.commit()
        return public(job)


def references(value, key):
    """Extract only explicit structured IDs; never substring-match message prose."""
    found = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k in {key, key + "s"}:
                for item in v if isinstance(v, list) else [v]:
                    if isinstance(item, (int, str)):
                        found.add(str(item))
            found.update(references(v, key))
    elif isinstance(value, list):
        for item in value:
            found.update(references(item, key))
    return found


def build_manifest(db, job):
    uid, cid = job.user_id, job.conversation_id
    runs = list(db.scalars(select(AgentRun).where(AgentRun.user_id == uid, AgentRun.conversation_id == cid)))
    run_ids = [r.id for r in runs]
    messages = list(db.scalars(select(AgentChatMessage).where(AgentChatMessage.user_id == uid, AgentChatMessage.conversation_id == cid)))
    other_metadata = list(db.scalars(select(AgentChatMessage.metadata_json).where(AgentChatMessage.user_id == uid, AgentChatMessage.conversation_id != cid)))
    other_metadata += list(db.scalars(select(AgentRun.graph_state).where(AgentRun.user_id == uid, AgentRun.conversation_id != cid)))
    skills = list(db.scalars(select(Skill).where(Skill.user_id == uid)))
    other_metadata += [{"context_recipe": s.context_recipe, "tool_plan": s.tool_plan} for s in skills]
    shared_docs, shared_artifacts = references(other_metadata, "document_id"), references(other_metadata, "artifact_id")
    referenced_paths = references(other_metadata, "file_path")
    doc_ids = references([m.metadata_json.get("attachments", []) for m in messages if m.metadata_json], "document_id")
    memory_refs = references([r.graph_state for r in runs], "memory_id")
    docs, memories, artifacts, research, paths, warnings = [], [], [], [], [], []
    for doc in db.scalars(select(Document).where(Document.user_id == uid, Document.id.in_([int(i) for i in doc_ids if i.isdigit()]))):
        if str(doc.id) in shared_docs or doc.source_type != "chat_upload":
            warnings.append({"kind": "document_retained", "id": doc.id})
            continue
        if doc.file_path and (doc.file_path in referenced_paths or db.scalar(select(Document.id).where(Document.file_path == doc.file_path, Document.id != doc.id).limit(1))):
            warnings.append({"kind": "document_shared_path", "id": doc.id})
            continue
        from src.web_app.services.document_ingest_task_manager import document_ingest_task_manager
        if doc.status in {"processing", "ingesting", "uploaded", "pending"} or document_ingest_task_manager.is_running(doc.id):
            raise DeletionError("DOCUMENT_BUSY", "附件仍在处理，请完成后重试删除。")
        docs.append(doc.id)
        if doc.file_path:
            paths.append({"kind": "document", "id": doc.id, "path": doc.file_path})
        doc.status = "deleting"
    for memory in db.scalars(select(Memory).where(Memory.user_id == uid)):
        meta = memory.metadata_json or {}
        if memory.memory_type in {"semantic", "episodic"} or meta.get("visible_in_long_term_memory") is True:
            continue
        owned = meta.get("conversation_id") == cid or str(meta.get("run_id", "")) in set(map(str, run_ids)) or str(memory.id) in memory_refs
        owned |= memory.source_type in {"conversation", "agent_conversation"} and memory.source_id == cid
        owned |= memory.source_type in {"run", "agent_run"} and memory.source_id in set(map(str, run_ids))
        if owned:
            memories.append(memory.id)
            memory.metadata_json = {**meta, "deletion_task_id": job.id, "status": "deleting"}
        elif not meta.get("conversation_id") and not meta.get("run_id") and not memory.source_id:
            warnings.append({"kind": "memory_owner_unknown", "id": memory.id})
    for item in db.scalars(select(ResearchRun).where(ResearchRun.user_id == uid, ResearchRun.agent_run_id.in_(run_ids))):
        if item.feed_card_id or item.skill_draft_id or (item.metadata_json or {}).get("independently_saved"):
            if item.artifact_id:
                shared_artifacts.add(str(item.artifact_id))
            warnings.append({"kind": "research_retained", "id": item.id})
        else:
            research.append(item.id)
    external_research = db.scalars(select(ResearchRun).where(ResearchRun.user_id == uid, ResearchRun.id.not_in(research)))
    shared_artifacts.update(str(r.artifact_id) for r in external_research if r.artifact_id)
    for artifact in db.scalars(select(Artifact).where(Artifact.user_id == uid, Artifact.run_id.in_(run_ids))):
        if str(artifact.id) in shared_artifacts or (artifact.metadata_json or {}).get("independently_saved") or artifact.public_url:
            warnings.append({"kind": "artifact_retained", "id": artifact.id})
            continue
        if artifact.file_path and (artifact.file_path in referenced_paths or db.scalar(select(Artifact.id).where(Artifact.file_path == artifact.file_path, Artifact.id != artifact.id).limit(1))):
            warnings.append({"kind": "artifact_shared_path", "id": artifact.id})
            continue
        artifacts.append(artifact.id)
        artifact.metadata_json = {**(artifact.metadata_json or {}), "deletion_task_id": job.id}
        if artifact.file_path:
            paths.append({"kind": "artifact", "id": artifact.id, "path": artifact.file_path})
    segments = list(db.scalars(select(AgentConversationSummarySegment.id).where(AgentConversationSummarySegment.user_id == uid,
        AgentConversationSummarySegment.conversation_id == cid)))
    other_threads = {str((value or {}).get("thread_id")) for value in db.scalars(select(AgentRun.graph_state).where(AgentRun.conversation_id != cid))}
    checkpoint_threads = {f"run:{i}" for i in run_ids}
    checkpoint_threads.update(str(r.graph_state["thread_id"]) for r in runs if (r.graph_state or {}).get("thread_id") and str(r.graph_state["thread_id"]) not in other_threads)
    return {"run_ids": run_ids, "document_ids": docs, "memory_ids": memories, "artifact_ids": artifacts,
        "research_ids": research, "segment_ids": segments, "paths": paths, "warnings": warnings,
        "checkpoint_threads": sorted(checkpoint_threads)}


def delete_sql(db, job):
    """No repository commits: caller commits final status and deletion together."""
    uid, cid, manifest = job.user_id, job.conversation_id, job.manifest
    ids = manifest["run_ids"]
    for model in (AgentConversationSummarySegment, AgentConversationSummary, AgentChatMessage):
        db.execute(delete(model).where(model.user_id == uid, model.conversation_id == cid))
    for model in (Approval, EvalRecord, AgentStep, AgentEvent, LLMCall, ToolCall):
        db.execute(delete(model).where(model.run_id.in_(ids)))
    db.execute(delete(AgentRunControl).where(AgentRunControl.run_id.in_(ids)))
    db.execute(update(AgentRunControl).where(AgentRunControl.successor_run_id.in_(ids)).values(successor_run_id=None))
    db.execute(update(AgentRun).where(AgentRun.supersedes_run_id.in_(ids)).values(supersedes_run_id=None))
    db.execute(delete(ResearchRun).where(ResearchRun.id.in_(manifest["research_ids"]), ResearchRun.user_id == uid))
    db.execute(update(ResearchRun).where(ResearchRun.agent_run_id.in_(ids)).values(agent_run_id=None))
    db.execute(delete(Artifact).where(Artifact.id.in_(manifest["artifact_ids"]), Artifact.user_id == uid))
    db.execute(update(Artifact).where(Artifact.run_id.in_(ids)).values(run_id=None))
    db.execute(update(AgentConversation).where(AgentConversation.last_run_id.in_(ids)).values(last_run_id=None))
    db.execute(delete(AgentRun).where(AgentRun.id.in_(ids), AgentRun.user_id == uid))
    db.execute(delete(Memory).where(Memory.user_id == uid, Memory.id.in_(manifest["memory_ids"])))
    db.execute(delete(DocumentChunk).where(DocumentChunk.user_id == uid, DocumentChunk.document_id.in_(manifest["document_ids"])))
    db.execute(delete(Document).where(Document.user_id == uid, Document.id.in_(manifest["document_ids"])))
    db.execute(delete(AgentConversation).where(AgentConversation.user_id == uid, AgentConversation.conversation_id == cid))


def cleanup_external(job):
    from src.web_app.services.deletion_resources import cleanup_resources
    cleanup_resources(job.user_id, job.conversation_id, job.manifest)


def perform(job_id, count_attempt=True):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job or job.status in TERMINAL:
            return
        if count_attempt:
            job.attempts += 1
        job.status = "running"
        job.phase = "resources"
        db.commit()
        try:
            if not job.manifest:
                with asset_lock:
                    job.manifest = build_manifest(db, job)
                    db.commit()
            cleanup_external(job)
            job.progress = {**(job.progress or {}), "external": True}
            db.commit()
            delete_sql(db, job)
            warnings = job.manifest.get("warnings", [])
            job.result = {"counts": {k: len(v) for k, v in job.manifest.items() if k.endswith("_ids")}, "warnings": warnings}
            job.manifest = {}
            job.error_message = ""
            job.status = "completed_with_warnings" if warnings else "completed"
            job.phase = "completed"
            db.commit()
        except Exception as exc:
            db.rollback()
            job = db.get(Job, job_id)
            job.status = "failed"
            job.error_message = f"{type(exc).__name__}: 清理未完成，请检查服务日志并重试。"
            db.commit()
            import logging
            logging.getLogger(__name__).exception("conversation deletion failed job_id=%s", job_id)


class DeletionManager:
    def __init__(self):
        self.tasks = {}

    def start(self, job_id):
        if job_id not in self.tasks:
            task = asyncio.create_task(self.run(job_id))
            self.tasks[job_id] = task
            task.add_done_callback(lambda _: self.tasks.pop(job_id, None))

    async def run(self, job_id):
        while True:
            with SessionLocal() as db:
                job = db.get(Job, job_id)
                if not job or job.status in TERMINAL or job.attempts >= 4:
                    return
                job.attempts += 1
                job.status, job.phase = "running", "stopping"
                db.commit()
            try:
                await self.run_attempt(job_id)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("deletion preparation failed job_id=%s", job_id)
                with SessionLocal() as db:
                    job = db.get(Job, job_id)
                    job.status, job.error_message = "failed", "删除准备失败，请检查服务日志并重试。"
                    db.commit()
            with SessionLocal() as db:
                job = db.get(Job, job_id)
                if job.status != "failed" or job.attempts >= 4:
                    return
            await asyncio.sleep(2)

    async def run_attempt(self, job_id):
        from src.web_app.services.agent_run_task_manager import agent_run_task_manager
        from src.web_app.services.summary_tasks import summary_tasks
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            uid, cid = job.user_id, job.conversation_id
            runs = list(db.scalars(select(AgentRun).where(AgentRun.user_id == uid, AgentRun.conversation_id == cid)))
            for run in runs:
                token = agent_run_task_manager._tokens.get(run.id)
                task = agent_run_task_manager._tasks.get(run.id)
                if task and not task.done():
                    if run.chat_control_phase != "enabled":
                        job.status, job.error_message = "failed", "任务仍在执行，结束后重试。"
                        db.commit()
                        return
                    if token:
                        token.cancelled = True
                    task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(task), 2)
                    except asyncio.CancelledError:
                        pass
                    except TimeoutError:
                        job.status, job.error_message = "failed", "停止超时，请等待任务退出后重试。"
                        db.commit()
                        return
        summary_tasks.pending.pop((uid, cid), None)
        summary = summary_tasks.tasks.get((uid, cid))
        if summary:
            try:
                await asyncio.wait_for(asyncio.shield(summary), 30)
            except TimeoutError:
                with SessionLocal() as db:
                    job = db.get(Job, job_id)
                    job.status, job.error_message = "failed", "摘要仍在收尾，请稍后重试。"
                    db.commit()
                return
        await asyncio.to_thread(perform, job_id, False)

    async def shutdown(self):
        if self.tasks:
            await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)


deletion_manager = DeletionManager()
