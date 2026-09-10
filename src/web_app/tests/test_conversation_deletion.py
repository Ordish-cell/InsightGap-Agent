import os
from uuid import uuid4
from types import SimpleNamespace
import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import sessionmaker
from src.web_app.db.base import Base
from src.web_app.models.orm import (User, AgentConversation, AgentRun, AgentChatMessage, AgentConversationSummary,
    AgentConversationSummarySegment, Memory, Document, DocumentChunk, EvalRecord, ConversationDeletionTask as Job)
from src.web_app.services import conversation_deletion as service


@pytest.fixture(params=["sqlite", "postgres"])
def data(request, tmp_path, monkeypatch):
    schema = None
    if request.param == "postgres":
        if os.getenv("DELETION_POSTGRES_TEST") != "1":
            pytest.skip("isolated PostgreSQL opt-in")
        from src.web_app.db.session import SessionLocal
        with SessionLocal() as source:
            url = source.get_bind().url
        admin = create_engine(url)
        schema = "test_deletion_" + uuid4().hex
        with admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'delete.db'}", connect_args={"check_same_thread": False})
        @event.listens_for(engine, "connect")
        def foreign_keys(conn, _):
            conn.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    monkeypatch.setattr(service, "SessionLocal", factory)
    monkeypatch.setattr(service, "cleanup_external", lambda job: None)
    with factory() as db:
        user = User(email="deletion@example.test", hashed_password="x")
        db.add(user); db.flush()
        run = AgentRun(user_id=user.id, conversation_id="target", status="completed", user_input="test")
        db.add(run); db.flush()
        db.add(AgentConversation(user_id=user.id, conversation_id="target", last_run_id=run.id))
        db.add(AgentConversation(user_id=user.id, conversation_id="other"))
        message = AgentChatMessage(user_id=user.id, conversation_id="target", run_id=run.id, message_id="m", role="user", content="test")
        db.add(message); db.flush()
        db.add(AgentConversationSummary(user_id=user.id, conversation_id="target", last_message_id=message.id))
        db.add(AgentConversationSummarySegment(user_id=user.id, conversation_id="target", start_message_id=message.id, end_message_id=message.id))
        db.add(EvalRecord(user_id=user.id, run_id=run.id, target_type="run"))
        for name, kind, meta in [("temporary", "working", {"conversation_id": "target"}),
            ("long", "semantic", {"conversation_id": "target"}), ("episode", "episodic", {}),
            ("visible", "working", {"conversation_id": "target", "visible_in_long_term_memory": True}),
            ("other", "working", {"conversation_id": "other"}), ("unknown", "working", {})]:
            db.add(Memory(user_id=user.id, content=name, memory_type=kind, metadata_json=meta))
        db.commit()
        uid, rid = user.id, run.id
    try:
        yield SimpleNamespace(factory=factory, user=uid, run=rid)
    finally:
        engine.dispose()
        if schema:
            assert schema.startswith("test_deletion_") and len(schema) == len("test_deletion_") + 32
            with admin.begin() as conn:
                conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            admin.dispose()


def test_delete_resolves_actual_foreign_keys_and_preserves_long_term(data):
    with data.factory() as db:
        accepted = service.request_deletion(db, data.user, "target")
        assert service.request_deletion(db, data.user, "target")["id"] == accepted["id"]
    service.perform(accepted["id"])
    with data.factory() as db:
        job = db.get(Job, accepted["id"])
        assert job.status == "completed_with_warnings", job.error_message
        assert not job.manifest
        assert db.scalar(select(AgentConversation).where(AgentConversation.conversation_id == "target")) is None
        assert db.get(AgentRun, data.run) is None
        assert not list(db.scalars(select(AgentChatMessage)))
        assert {m.content for m in db.scalars(select(Memory))} == {"long", "episode", "visible", "other", "unknown"}
        assert db.scalar(select(AgentConversation).where(AgentConversation.conversation_id == "other"))
        assert service.request_deletion(db, data.user, "target")["id"] == job.id


def test_external_failure_is_retryable_and_does_not_delete_sql(data, monkeypatch):
    with data.factory() as db:
        job_id = service.request_deletion(db, data.user, "target")["id"]
    monkeypatch.setattr(service, "cleanup_external", lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    service.perform(job_id)
    with data.factory() as db:
        assert db.get(Job, job_id).status == "failed"
        assert db.get(Job, job_id).manifest
        assert db.get(AgentRun, data.run)
    monkeypatch.setattr(service, "cleanup_external", lambda _: None)
    service.perform(job_id)
    with data.factory() as db:
        assert db.get(Job, job_id).status == "completed_with_warnings"


def test_sql_failure_rolls_back_all_business_deletes(data, monkeypatch):
    with data.factory() as db:
        job_id = service.request_deletion(db, data.user, "target")["id"]
    original = service.delete_sql
    def fail(db, job):
        original(db, job)
        raise RuntimeError("commit stage fault")
    monkeypatch.setattr(service, "delete_sql", fail)
    service.perform(job_id)
    with data.factory() as db:
        assert db.get(AgentRun, data.run)
        assert db.scalar(select(AgentConversationSummary))
        assert db.scalar(select(AgentChatMessage))
        assert db.get(Job, job_id).status == "failed"


def test_scope_and_running_research_guard(data):
    with data.factory() as db:
        with pytest.raises(service.DeletionError) as exc:
            service.request_deletion(db, data.user + 1, "target")
        assert exc.value.status == 404
        run = db.get(AgentRun, data.run)
        run.status, run.chat_control_phase = "running", "disabled"
        db.commit()
        with pytest.raises(service.DeletionError) as exc:
            service.request_deletion(db, data.user, "target")
        assert exc.value.code == "CONVERSATION_BUSY"
        assert not list(db.scalars(select(Job)))


def test_shared_document_is_not_owned_by_deletion(data):
    with data.factory() as db:
        doc = Document(user_id=data.user, filename="shared.txt", file_path="unused", source_type="chat_upload", status="ingested")
        db.add(doc); db.flush()
        for cid in ("target", "other"):
            db.add(AgentChatMessage(user_id=data.user, conversation_id=cid, message_id=cid, role="user", metadata_json={"attachments": [{"document_id": doc.id}]}))
        db.commit()
        job_id = service.request_deletion(db, data.user, "target")["id"]
        manifest = service.build_manifest(db, db.get(Job, job_id))
        assert manifest["document_ids"] == []


def test_file_path_cannot_escape_storage(tmp_path, monkeypatch):
    from src.web_app.services.deletion_resources import safe_file
    monkeypatch.setattr("src.web_app.core.config.settings.artifact_storage_path", str(tmp_path / "owned"))
    with pytest.raises(ValueError):
        safe_file(str(tmp_path / "elsewhere.txt"), "artifact")
    with pytest.raises(ValueError):
        safe_file(str(tmp_path / "owned"), "artifact")


@pytest.mark.parametrize("backend", ["local", "server"])
def test_real_qdrant_scoped_cleanup_preserves_other_users_and_memory(monkeypatch, backend):
    if backend == "server" and os.getenv("DELETION_QDRANT_TEST") != "1":
        pytest.skip("isolated Qdrant server opt-in")
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    from src.web_app.core.config import settings
    from src.web_app.services.deletion_resources import cleanup_resources
    prefix = "test_deletion_" + uuid4().hex
    client = QdrantClient(":memory:") if backend == "local" else QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=10)
    original_close = client.close
    if backend == "local":
        monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr(settings, "qdrant_url", "http://isolated-test")
    fields = [("qdrant_collection", "document_id", "1"), ("qdrant_hybrid_collection", "document_id", "1"),
              ("memory_qdrant_collection", "memory_id", "1"), ("conversation_segment_vector_collection", "conversation_id", "target")]
    collections = []
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", False)
    monkeypatch.setattr(settings, "enable_neo4j", False)
    try:
        for index, (setting, field, value) in enumerate(fields):
            collection = prefix + "_" + str(index)
            monkeypatch.setattr(settings, setting, collection)
            client.create_collection(collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
            collections.append(collection)
            client.upsert(collection, points=[PointStruct(id=1, vector=[1., 0.], payload={"user_id": "7", field: value}),
                PointStruct(id=2, vector=[1., 0.], payload={"user_id": "8", field: value}),
                PointStruct(id=3, vector=[1., 0.], payload={"user_id": "7", field: "other"})], wait=True)
        manifest = {"document_ids": [1], "memory_ids": [1], "segment_ids": [1], "paths": [], "checkpoint_threads": []}
        cleanup_resources(7, "target", manifest)
        cleanup_resources(7, "target", manifest)
        for collection in collections:
            remaining, _ = client.scroll(collection, limit=10)
            assert {p.id for p in remaining} == {2, 3}
    finally:
        for collection in collections:
            assert collection.startswith(prefix + "_")
            client.delete_collection(collection)
        original_close()


@pytest.mark.asyncio
async def test_manager_waits_for_chat_and_summary_before_resources(data, monkeypatch):
    import asyncio
    from src.web_app.agent.runtime.chat_control import ChatExecution
    from src.web_app.services.agent_run_task_manager import agent_run_task_manager
    from src.web_app.services.summary_tasks import summary_tasks
    stopped = asyncio.Event()
    async def chat():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    task = asyncio.create_task(chat())
    await asyncio.sleep(0)
    monkeypatch.setattr(agent_run_task_manager, "_tasks", {data.run: task})
    monkeypatch.setattr(agent_run_task_manager, "_tokens", {data.run: ChatExecution(data.run)})
    summary_done = []
    async def summary():
        await asyncio.sleep(.02)
        summary_done.append(True)
    summary_task = asyncio.create_task(summary())
    monkeypatch.setattr(summary_tasks, "tasks", {(data.user, "target"): summary_task})
    def cleanup(job):
        assert stopped.is_set() and summary_done
    monkeypatch.setattr(service, "cleanup_external", cleanup)
    with data.factory() as db:
        run = db.get(AgentRun, data.run)
        run.status, run.chat_control_phase = "running", "enabled"
        db.commit()
        job_id = service.request_deletion(db, data.user, "target")["id"]
    await service.DeletionManager().run(job_id)
    with data.factory() as db:
        assert db.get(Job, job_id).status in service.TERMINAL


def test_migration_missing_blocks_delete(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'unmigrated.db'}")
    with sessionmaker(engine)() as db:
        with pytest.raises(service.DeletionError) as exc:
            service.request_deletion(db, 1, "target")
        assert exc.value.code == "DELETION_MIGRATION_REQUIRED"
    engine.dispose()


@pytest.mark.asyncio
async def test_preparation_failures_have_bounded_retries(data, monkeypatch):
    import asyncio
    manager = service.DeletionManager()
    calls = []
    async def failing_attempt(job_id):
        calls.append(job_id)
        raise RuntimeError("injected preparation failure")
    async def no_delay(_):
        pass
    monkeypatch.setattr(manager, "run_attempt", failing_attempt)
    monkeypatch.setattr(asyncio, "sleep", no_delay)
    with data.factory() as db:
        job_id = service.request_deletion(db, data.user, "target")["id"]
    await manager.run(job_id)
    await manager.run(job_id)
    with data.factory() as db:
        job = db.get(Job, job_id)
        assert job.status == "failed" and job.attempts == 4
        assert len(calls) == 4
        assert db.get(AgentRun, data.run) is not None


def test_pending_approval_is_cancelled_without_execution(data):
    from src.web_app.models.orm import Approval
    from src.web_app.services.deletion_guard import check_conversation, ConversationDeletingError
    with data.factory() as db:
        run = db.get(AgentRun, data.run)
        run.status = "waiting_approval"
        approval = Approval(user_id=data.user, run_id=data.run, approval_type="tool", title="pending")
        db.add(approval); db.commit()
        with pytest.raises(service.DeletionError):
            service.request_deletion(db, data.user, "target")
        job = service.request_deletion(db, data.user, "target", cancel_pending=True)
        db.refresh(approval)
        assert approval.status == "cancelled"
        with pytest.raises(ConversationDeletingError):
            check_conversation(db, data.user, "target")
        assert service.request_deletion(db, data.user, "target")["id"] == job["id"]


def test_owned_file_cleanup_is_idempotent(tmp_path, monkeypatch):
    from src.web_app.core.config import settings
    from src.web_app.services.deletion_resources import cleanup_resources
    monkeypatch.setattr(settings, "artifact_storage_path", str(tmp_path))
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", False)
    monkeypatch.setattr(settings, "enable_neo4j", False)
    owned, other = tmp_path / "owned.txt", tmp_path / "other.txt"
    owned.write_text("owned"); other.write_text("other")
    manifest = {"document_ids": [], "memory_ids": [], "segment_ids": [], "checkpoint_threads": [],
                "paths": [{"path": str(owned), "kind": "artifact"}]}
    cleanup_resources(7, "target", manifest)
    cleanup_resources(7, "target", manifest)
    assert not owned.exists() and other.read_text() == "other"


def test_incremental_migration_creates_only_job_table(data, monkeypatch):
    import importlib
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("deletion_migration", Path("alembic/versions/20260909_0015_conversation_deletion.py"))
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with data.factory() as db:
        engine = db.get_bind()
    with engine.begin() as conn:
        Job.__table__.drop(conn)
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(conn)))
        migration.upgrade()
    with data.factory() as db:
        assert db.get(AgentRun, data.run) is not None
        assert service.request_deletion(db, data.user, "target")["status"] == "pending"


def test_api_jobs_restore_owner_state_and_manual_retry(data, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.web_app.api.v1 import agent
    from src.web_app.main import deletion_error_handler
    app = FastAPI()
    app.include_router(agent.router, prefix="/agent")
    app.add_exception_handler(service.DeletionError, deletion_error_handler)
    identity = {"id": data.user}
    def db_dependency():
        with data.factory() as db:
            yield db
    app.dependency_overrides[agent.get_db] = db_dependency
    app.dependency_overrides[agent.get_current_user_id] = lambda: identity["id"]
    started = []
    monkeypatch.setattr(service.deletion_manager, "start", started.append)
    client = TestClient(app)
    first = client.delete("/agent/conversations/target/hard")
    assert first.status_code == 202
    job_id = first.json()["data"]["id"]
    assert client.delete("/agent/conversations/target/hard").json()["data"]["id"] == job_id
    assert client.get("/agent/deletion-tasks").json()["data"][0]["id"] == job_id
    identity["id"] = data.user + 999
    assert client.get(f"/agent/deletion-tasks/{job_id}").status_code == 404
    assert client.post(f"/agent/deletion-tasks/{job_id}/retry").status_code == 404
    assert client.delete("/agent/conversations/target/hard").status_code == 404
    identity["id"] = data.user
    with data.factory() as db:
        job = db.get(Job, job_id)
        job.status, job.attempts = "failed", 4
        db.commit()
    result = client.post(f"/agent/deletion-tasks/{job_id}/retry")
    assert result.status_code == 202
    assert result.json()["data"]["attempts"] == 0
    assert "manifest" not in result.json()["data"]


def test_reserved_memory_rejects_stale_writer(data):
    from src.web_app.db.repositories.memory_repository import MemoryRepository
    from src.web_app.services.deletion_guard import ConversationDeletingError
    with data.factory() as stale, data.factory() as db:
        memory = stale.scalar(select(Memory).where(Memory.content == "temporary"))
        job_id = service.request_deletion(db, data.user, "target")["id"]
        job = db.get(Job, job_id)
        job.manifest = service.build_manifest(db, job)
        db.commit()
        with pytest.raises(ConversationDeletingError):
            MemoryRepository(stale).update(memory, content="late update")


def test_exclusive_results_removed_and_saved_skill_preserved(data):
    from src.web_app.models.orm import Artifact, ResearchRun, Skill
    with data.factory() as db:
        exclusive = Artifact(user_id=data.user, run_id=data.run, artifact_type="report", title="exclusive")
        saved = Artifact(user_id=data.user, run_id=data.run, artifact_type="report", title="saved", metadata_json={"independently_saved": True})
        skill = Skill(user_id=data.user, name="reusable", status="active")
        db.add_all([exclusive, saved, skill]); db.flush()
        exclusive_id, saved_id, skill_id = exclusive.id, saved.id, skill.id
        db.add(ResearchRun(id="exclusive", user_id=data.user, agent_run_id=data.run, query="test", artifact_id=exclusive.id))
        db.add(ResearchRun(id="saved", user_id=data.user, agent_run_id=data.run, query="test", artifact_id=saved.id, skill_draft_id=skill.id))
        db.commit()
        job_id = service.request_deletion(db, data.user, "target")["id"]
    service.perform(job_id)
    with data.factory() as db:
        assert db.get(Job, job_id).status in service.TERMINAL
        assert db.get(Artifact, exclusive_id) is None
        assert db.get(ResearchRun, "exclusive") is None
        assert db.get(Artifact, saved_id).run_id is None
        assert db.get(ResearchRun, "saved").agent_run_id is None
        assert db.get(Skill, skill_id).status == "active"


def test_checkpoint_cleanup_uses_isolated_postgres_thread_ids(data, monkeypatch):
    import psycopg
    from src.web_app.core.config import settings
    from src.web_app.services.deletion_resources import cleanup_resources
    from src.web_app.agent.runtime import checkpoint_cleanup
    with data.factory() as db:
        engine = db.get_bind()
        if engine.dialect.name != "postgresql":
            pytest.skip("PostgreSQL checkpoint tables")
        schema = db.scalar(text("SELECT current_schema()"))
    assert schema.startswith("test_deletion_")
    for table in checkpoint_cleanup.CHECKPOINT_DATA_TABLES:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE TABLE "{table}" (thread_id text, payload text)'))
            conn.execute(text(f'INSERT INTO "{table}" VALUES (:target, :content), (:other, :content)'),
                         {"target": "run:target", "other": "run:other", "content": "test"})
    connection_string = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    original_connect = psycopg.connect
    monkeypatch.setattr(checkpoint_cleanup, "_pg_conn_string", lambda: connection_string)
    monkeypatch.setattr(psycopg, "connect", lambda info: original_connect(info, options=f"-csearch_path={schema}"))
    monkeypatch.setattr(settings, "agent_langgraph_checkpointer_enabled", True)
    monkeypatch.setattr(settings, "enable_neo4j", False)
    manifest = {"document_ids": [], "memory_ids": [], "segment_ids": [], "paths": [], "checkpoint_threads": ["run:target"]}
    cleanup_resources(data.user, "target", manifest)
    cleanup_resources(data.user, "target", manifest)
    with engine.connect() as conn:
        for table in checkpoint_cleanup.CHECKPOINT_DATA_TABLES:
            assert conn.execute(text(f'SELECT thread_id FROM "{table}"')).scalars().all() == ["run:other"]
