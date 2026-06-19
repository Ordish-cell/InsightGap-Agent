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


class TestHardDeleteCleansCheckpoints:
    def test_checkpoint_cleanup_called_before_orm_delete(self):
        """hard_delete_conversation calls delete_checkpoints_for_runs before ORM delete."""
        db = make_test_session()
        user = _user(db)
        conv, runs = _create_conversation_with_runs(
            db, user.id, "conv-test-1", ["completed", "completed"]
        )
        run_ids = [r.id for r in runs]

        with patch(
            "src.web_app.services.agent_service.delete_checkpoints_for_runs"
        ) as mock_delete:
            result = hard_delete_conversation(db, user.id, "conv-test-1")

            mock_delete.assert_called_once_with(run_ids)
            assert result["deleted_records"] >= 1

    def test_orm_data_deleted_after_checkpoint_cleanup(self):
        """ORM records are gone after hard_delete, even if checkpoint cleanup fails."""
        db = make_test_session()
        user = _user(db)
        conv, runs = _create_conversation_with_runs(
            db, user.id, "conv-rm-1", ["completed"]
        )
        run_id = runs[0].id

        with patch(
            "src.web_app.services.agent_service.delete_checkpoints_for_runs",
            side_effect=Exception("checkpoint DB down"),
        ):
            result = hard_delete_conversation(db, user.id, "conv-rm-1")

        assert result["deleted_records"] >= 1
        # Run should be gone
        run_repo = AgentRunRepository(db)
        assert run_repo.get_by_id(run_id) is None
        # Conversation should be gone
        assert db.get(AgentConversation, conv.id) is None

    def test_single_run_conversation(self):
        """Single-run conversation: checkpoint deleted, ORM deleted."""
        db = make_test_session()
        user = _user(db)
        conv, runs = _create_conversation_with_runs(
            db, user.id, "conv-single-1", ["completed"]
        )

        with patch(
            "src.web_app.services.agent_service.delete_checkpoints_for_runs"
        ) as mock_delete:
            result = hard_delete_conversation(db, user.id, "conv-single-1")

            mock_delete.assert_called_once_with([1])
            assert result["deleted_records"] >= 1


class TestHardDeletePendingGuard:
    def test_blocked_when_pending_approval_exists(self):
        """409 when conversation has a pending approval and cancel_pending=False."""
        db = make_test_session()
        user = _user(db)
        _create_blocked_conversation(db, user.id, "conv-blocked-1")

        with pytest.raises(PendingApprovalExistsError) as exc_info:
            hard_delete_conversation(db, user.id, "conv-blocked-1", cancel_pending=False)

        assert "等待审批" in str(exc_info.value)

    def test_cancel_pending_cleans_checkpoints(self):
        """cancel_pending=True cancels approvals AND cleans checkpoints."""
        db = make_test_session()
        user = _user(db)
        _create_blocked_conversation(db, user.id, "conv-cancel-1")

        with patch(
            "src.web_app.services.agent_service.delete_checkpoints_for_runs"
        ) as mock_delete:
            result = hard_delete_conversation(
                db, user.id, "conv-cancel-1", cancel_pending=True
            )

            mock_delete.assert_called_once_with([1])
            assert result["cancelled_approvals"] >= 1
            assert result["cancelled_runs"] >= 1

    def test_cancel_pending_approval_becomes_cancelled(self):
        """After cancel_pending=True, approval status is 'cancelled'."""
        db = make_test_session()
        user = _user(db)
        _, _, approval, _, _ = _create_blocked_conversation(
            db, user.id, "conv-cancel-2"
        )

        with patch(
            "src.web_app.services.agent_service.delete_checkpoints_for_runs"
        ):
            hard_delete_conversation(db, user.id, "conv-cancel-2", cancel_pending=True)

        db.expire_all()
        a = db.get(Approval, approval.id)
        assert a is not None
        assert a.status == "cancelled"

    def test_no_pending_approval_allows_delete(self):
        """Conversation without pending approvals is deleted directly."""
        db = make_test_session()
        user = _user(db)
        _create_conversation_with_runs(db, user.id, "conv-clean-1", ["completed"])

        with patch(
            "src.web_app.services.agent_service.delete_checkpoints_for_runs"
        ) as mock_delete:
            result = hard_delete_conversation(db, user.id, "conv-clean-1")

            mock_delete.assert_called_once()
            assert result["deleted_records"] >= 1


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
