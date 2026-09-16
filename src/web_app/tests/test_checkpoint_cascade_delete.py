"""Tests for checkpoint cascade deletion on hard_delete_conversation.

Verifies:
  1. hard_delete_conversation calls delete_checkpoints_for_runs
  2. Pending approval guard still blocks deletion
  3. cancel_pending=True path also cleans checkpoints
  4. delete_checkpoints_for_runs builds correct SQL
  5. cleanup_orphan_checkpoints finds and cleans orphans
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest

from src.web_app.agent.runtime.checkpoint_cleanup import (
    CHECKPOINT_DATA_TABLES,
    _thread_ids_from_run_ids,
    cleanup_orphan_checkpoints,
    delete_checkpoints_for_runs,
)
from src.web_app.db.repositories.agent_repository import AgentRunRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.models.orm import (
    AgentChatMessage,
    AgentConversation,
    AgentRun,
    Approval,
    ToolCall,
    User,
)
from src.web_app.services.agent_service import (
    PendingApprovalExistsError,
    hard_delete_conversation,
)
from src.web_app.tests.db_test_utils import make_test_session


# ── helpers ──────────────────────────────────────────────────────────


def _user(db, email="checkpoint-test@example.com"):
    u = User(email=email, hashed_password="x")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _create_conversation_with_runs(
    db,
    user_id: int,
    conversation_id: str = "conv-test-1",
    run_statuses: list[str] | None = None,
):
    """Create a conversation with associated runs."""
    if run_statuses is None:
        run_statuses = ["completed", "completed"]

    conv = AgentConversation(
        conversation_id=conversation_id,
        user_id=user_id,
        title="Test Conversation",
        source="agent_page",
        status="active",
        thread_id=f"user:{user_id}:conversation:{conversation_id}",
    )
    db.add(conv)
    db.flush()

    runs = []
    for i, status in enumerate(run_statuses):
        run_id = i + 1
        run = AgentRun(
            id=run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            thread_id=f"run:{run_id}",
            run_type="agent_runtime",
            mode="react",
            status=status,
            user_input=f"test input {run_id}",
            graph_state={"thread_id": f"run:{run_id}"},
        )
        db.add(run)
        runs.append(run)

    db.commit()
    return conv, runs


def _create_blocked_conversation(db, user_id: int, conversation_id: str = "conv-blocked-1"):
    """Create a conversation with a waiting_approval run and pending approval."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conv = AgentConversation(
        conversation_id=conversation_id,
        user_id=user_id,
        title="Blocked Conversation",
        source="agent_page",
        status="active",
        thread_id=f"user:{user_id}:conversation:{conversation_id}",
    )
    db.add(conv)
    db.flush()

    run = AgentRun(
        id=1,
        user_id=user_id,
        conversation_id=conversation_id,
        thread_id="run:1",
        run_type="agent_runtime",
        mode="react",
        status="waiting_approval",
        user_input="send an email",
        graph_state={
            "status": "waiting_approval",
            "approval_required": True,
            "pending_approval_id": "100",
            "pending_tool_name": "email.send",
            "thread_id": "run:1",
        },
    )
    db.add(run)
    db.flush()

    approval = Approval(
        id=100,
        user_id=user_id,
        run_id=1,
        approval_type="tool_approval",
        title="Test approval",
        description="Test",
        payload={"tool_name": "email.send"},
        status="pending",
    )
    db.add(approval)

    msg = AgentChatMessage(
        message_id=str(uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        run_id=1,
        thread_id="run:1",
        role="assistant",
        content="",
        status="waiting_approval",
        metadata_json={"run_id": 1},
    )
    db.add(msg)

    tc = ToolCall(
        id=100,
        user_id=user_id,
        run_id=1,
        tool_name="email.send",
        input={"to": "test@test.com"},
        status="waiting_approval",
    )
    db.add(tc)
    db.commit()

    return conv, run, approval, msg, tc


# ── unit tests: helpers ──────────────────────────────────────────────


class TestThreadIdFromRunIds:
    def test_single(self):
        assert _thread_ids_from_run_ids([42]) == ["run:42"]

    def test_multiple(self):
        assert _thread_ids_from_run_ids([1, 2, 3]) == ["run:1", "run:2", "run:3"]

    def test_empty(self):
        assert _thread_ids_from_run_ids([]) == []


# ── unit tests: delete_checkpoints_for_runs ──────────────────────────


class TestDeleteCheckpointsForRuns:
    def test_empty_run_ids_returns_zero(self):
        assert delete_checkpoints_for_runs([]) == 0

    def test_deletes_from_all_three_tables(self):
        """Verify correct SQL is executed against all 3 checkpoint tables."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 5
        mock_conn.cursor.return_value = mock_cur

        with patch("psycopg.connect", return_value=mock_conn) as mock_connect:
            result = delete_checkpoints_for_runs([1, 2, 3])

            assert result == 15  # 5 rows x 3 tables
            assert mock_cur.execute.call_count == 3
            for table in CHECKPOINT_DATA_TABLES:
                mock_cur.execute.assert_any_call(
                    f"DELETE FROM {table} WHERE thread_id = ANY(%s)",
                    (["run:1", "run:2", "run:3"],),
                )
            mock_conn.commit.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_no_conn_string_returns_zero(self):
        with patch(
            "src.web_app.agent.runtime.checkpoint_cleanup._pg_conn_string",
            return_value="",
        ):
            assert delete_checkpoints_for_runs([1, 2]) == 0


# ── integration tests: hard_delete_conversation ─────────────────────


def _perform(db, job_id, cleanup):
    from sqlalchemy.orm import sessionmaker
    from src.web_app.services import conversation_deletion as service
    from src.web_app.models.orm import ConversationDeletionTask
    with patch.object(service, "SessionLocal", sessionmaker(bind=db.get_bind())), patch.object(service, "cleanup_external", cleanup):
        service.perform(job_id)
    db.expire_all()
    return db.get(ConversationDeletionTask, job_id)


class TestHardDeleteCleansCheckpoints:
    def test_checkpoint_cleanup_called_before_orm_delete(self):
        db = make_test_session()
        user = _user(db)
        conv, runs = _create_conversation_with_runs(db, user.id)
        ids = [r.id for r in runs]
        checked = []
        def cleanup(job):
            assert set(job.manifest["checkpoint_threads"]) >= {f"run:{i}" for i in ids}
            assert all(db.get(AgentRun, i) is not None for i in ids)
            checked.append(True)
        result = hard_delete_conversation(db, user.id, conv.conversation_id)
        assert result["status"] == "pending"
        job = _perform(db, result["id"], cleanup)
        assert checked == [True] and job.status == "completed"
        assert all(db.get(AgentRun, i) is None for i in ids)

    def test_orm_data_retained_when_checkpoint_cleanup_fails(self):
        db = make_test_session()
        user = _user(db)
        conv, runs = _create_conversation_with_runs(db, user.id, "retain", ["completed"])
        cid, rid = conv.id, runs[0].id
        result = hard_delete_conversation(db, user.id, "retain")
        def unavailable(job): raise RuntimeError("checkpoint DB down")
        job = _perform(db, result["id"], unavailable)
        assert job.status == "failed" and job.manifest
        assert db.get(AgentRun, rid) is not None and db.get(AgentConversation, cid) is not None
        assert _perform(db, result["id"], lambda job: None).status == "completed"
        assert db.get(AgentRun, rid) is None

    def test_single_run_conversation(self):
        db = make_test_session()
        user = _user(db)
        conv, runs = _create_conversation_with_runs(db, user.id, "single", ["completed"])
        cid, rid = conv.id, runs[0].id
        scopes = []
        result = hard_delete_conversation(db, user.id, "single")
        job = _perform(db, result["id"], lambda job: scopes.extend(job.manifest["checkpoint_threads"]))
        assert job.status == "completed" and f"run:{rid}" in scopes
        assert db.get(AgentConversation, cid) is None


class TestHardDeletePendingGuard:
    def test_blocked_when_pending_approval_exists(self):
        from src.web_app.services.conversation_deletion import DeletionError
        db = make_test_session()
        user = _user(db)
        _create_blocked_conversation(db, user.id, "blocked")
        with pytest.raises(DeletionError) as exc:
            hard_delete_conversation(db, user.id, "blocked", cancel_pending=False)
        assert exc.value.code == "CONVERSATION_HAS_PENDING_APPROVAL"

    def test_cancel_pending_cleans_checkpoints(self):
        db = make_test_session()
        user = _user(db)
        _create_blocked_conversation(db, user.id, "cancel")
        result = hard_delete_conversation(db, user.id, "cancel", cancel_pending=True)
        scopes = []
        job = _perform(db, result["id"], lambda job: scopes.extend(job.manifest["checkpoint_threads"]))
        assert job.status == "completed" and "run:1" in scopes
        assert db.get(AgentRun, 1) is None

    def test_cancel_pending_approval_becomes_cancelled(self):
        db = make_test_session()
        user = _user(db)
        _, _, approval, _, _ = _create_blocked_conversation(db, user.id, "cancel-approval")
        aid = approval.id
        hard_delete_conversation(db, user.id, "cancel-approval", cancel_pending=True)
        db.expire_all()
        assert db.get(Approval, aid).status == "cancelled"

    def test_no_pending_approval_allows_delete(self):
        db = make_test_session()
        user = _user(db)
        _create_conversation_with_runs(db, user.id, "clean", ["completed"])
        result = hard_delete_conversation(db, user.id, "clean")
        assert result["status"] == "pending"
        assert _perform(db, result["id"], lambda job: None).status == "completed"


# ── orphan cleanup tests ─────────────────────────────────────────────


class TestCleanupOrphanCheckpoints:
    def test_no_conn_string_returns_error(self):
        with patch(
            "src.web_app.agent.runtime.checkpoint_cleanup._pg_conn_string",
            return_value="",
        ):
            result = cleanup_orphan_checkpoints()
            assert "no database connection string" in result["errors"]

    def test_dry_run_counts_orphans_without_deleting(self):
        """dry_run=True counts orphans but doesn't commit."""
        mock_rows = [("run:1",), ("run:2",), ("run:999",)]
        mock_conn = MagicMock()
        mock_cur = MagicMock()

        # First query: all thread_ids
        # For "run:1" -> agent_runs exists; "run:2" -> exists; "run:999" -> None = orphan
        fetchone_sequence = [("exists",), ("exists",), None]

        # For dry_run counting: 3 rows returned per COUNT query
        count_sequence = [(3,)]

        fetchone_call = {"count": 0}

        def fetchall_side_effect():
            return mock_rows

        def fetchone_side_effect():
            idx = fetchone_call["count"]
            fetchone_call["count"] += 1
            if idx < len(fetchone_sequence):
                return fetchone_sequence[idx]
            return count_sequence[min(idx - len(fetchone_sequence), len(count_sequence) - 1)]

        mock_cur.fetchall = MagicMock(side_effect=fetchall_side_effect)
        mock_cur.fetchone = MagicMock(side_effect=fetchone_side_effect)
        mock_cur.rowcount = 0
        mock_conn.cursor.return_value = mock_cur

        with patch("psycopg.connect", return_value=mock_conn):
            result = cleanup_orphan_checkpoints(dry_run=True)

        assert len(result["orphan_thread_ids"]) == 1
        assert "run:999" in result["orphan_thread_ids"]
        assert result["dry_run"] is True
        mock_conn.commit.assert_not_called()

    def test_empty_checkpoints_returns_early(self):
        """No run-prefixed thread_ids → early return, no errors."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur

        with patch("psycopg.connect", return_value=mock_conn):
            result = cleanup_orphan_checkpoints()

        assert result["orphan_thread_ids"] == []
        assert result["deleted_rows"] == 0
        assert result["errors"] == []
