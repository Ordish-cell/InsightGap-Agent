from __future__ import annotations

import asyncio
import logging

from src.web_app.db.repositories.document_repository import DocumentRepository
from src.web_app.db.session import SessionLocal

logger = logging.getLogger(__name__)


class DocumentIngestTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[int, asyncio.Task[None]] = {}

    def is_running(self, document_id: int) -> bool:
        task = self._tasks.get(document_id)
        return bool(task and not task.done())

    def start(self, user_id: int, document_id: int) -> bool:
        if self.is_running(document_id):
            return False
        task = asyncio.create_task(self._run(user_id, document_id))
        self._tasks[document_id] = task
        task.add_done_callback(lambda completed, did=document_id: self._discard(did, completed))
        return True

    async def _run(self, user_id: int, document_id: int) -> None:
        try:
            await asyncio.to_thread(self._run_sync, user_id, document_id)
        except asyncio.CancelledError:
            await asyncio.to_thread(self._mark_interrupted, user_id, document_id)
            raise
        except Exception:
            logger.exception("document.background_ingest_failed user_id=%s document_id=%s", user_id, document_id)

    def _run_sync(self, user_id: int, document_id: int) -> None:
        from src.web_app.services.document_service import document_service

        with SessionLocal() as db:
            document_service.ingest_chat_document(db, user_id, document_id)

    def _mark_interrupted(self, user_id: int, document_id: int) -> None:
        with SessionLocal() as db:
            repo = DocumentRepository(db)
            document = repo.get_by_id_for_user(user_id, document_id)
            if document and document.status in {"processing", "ingesting", "uploaded"}:
                repo.mark_failed(document, "Application shutdown interrupted document ingestion", failed_stage="interrupted")

    def _discard(self, document_id: int, task: asyncio.Task[None]) -> None:
        if self._tasks.get(document_id) is task:
            self._tasks.pop(document_id, None)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


document_ingest_task_manager = DocumentIngestTaskManager()
