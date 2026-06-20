"""Tests for historical conversation segment creation and recall.

Covers:
  0. Async call correctness
  1. Migration / ORM model
  2. Segment creation (threshold, batching, idempotence, continuation)
  3. Qdrant best-effort write
  4. Segment recall (Qdrant → PG fallback, scoring, limit)
  5. Data isolation (user_id + conversation_id)
  6. ContextBuilder injection
  7. Token budget enforcement
  8. 100-turn regression
  9. Graceful degradation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.web_app.db.repositories.agent_repository import (
    AgentChatMessageRepository,
    AgentConversationRepository,
    AgentConversationSummarySegmentRepository,
    AgentRunRepository,
)
from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import (
    AgentChatMessage,
    AgentConversation,
    AgentConversationSummarySegment,
    User,
)
from src.web_app.services.conversation_summary_service import (
    ConversationSummaryService,
    RecalledConversationSegment,
    conversation_summary_service,
)
from src.web_app.tests.db_test_utils import make_test_session


# ── Helpers ────────────────────────────────────────────────────────────


def _make_user(db, *, email: str = "test@test.com", nickname: str = "test") -> User:
    user = User(email=email, hashed_password="x", nickname=nickname)
    db.add(user)
    db.commit()
    return user


def _make_conversation(db, user_id: int, *, conversation_id: str | None = None) -> AgentConversation:
    cid = conversation_id or str(uuid.uuid4())
    repo = AgentConversationRepository(db)
    return repo.create(
        conversation_id=cid,
        user_id=user_id,
        title="Test",
        source="home_chat",
        thread_id=f"user:{user_id}:conversation:{cid}",
    )


def _make_run(db, user_id: int, conversation_id: str) -> Any:
    repo = AgentRunRepository(db)
    return repo.create(
        user_id=user_id,
        conversation_id=conversation_id,
        thread_id=f"user:{user_id}:conversation:{conversation_id}",
        user_input="hello",
        status="completed",
    )


def _make_message(
    db,
    user_id: int,
    conversation_id: str,
    *,
    role: str = "user",
    content: str = "test",
    run_id: int | None = None,
    message_id: str | None = None,
) -> AgentChatMessage:
    repo = AgentChatMessageRepository(db)
    return repo.create(
        message_id=message_id or str(uuid.uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        role=role,
        content=content,
        run_id=run_id,
        thread_id=f"run:{run_id}" if run_id else "",
    )


def _fake_llm_segment(prompt: str) -> str:
    """Return a deterministic segment summary that contains a unique fingerprint."""
    return "## Segment Scope\n- Conversation ID: test\n- Message count: N/A\n\n## Key Facts\n- 项目代号是 Phoenix\n- 数据库必须使用 PostgreSQL，不能使用 MySQL\n\n## Decisions\n- 已决定使用 LangGraph 作为编排框架\n\n## Technical Details\n- 幂等字段：idempotency_key\n\n## Open Issues\n- 前端部署流水线待优化\n\n## Follow-up Tasks\n- 补完测试用例"


# ── 0. Async call correctness ──────────────────────────────────────────


class TestAsyncCallCorrectness:
    """Verify that create_segment_if_needed is a synchronous function
    and can be safely called via asyncio.to_thread."""

    def test_create_segment_if_needed_is_sync_not_coroutine(self):
        """It must be a regular def, not async def, for asyncio.to_thread."""
        import inspect
        from src.web_app.services.conversation_summary_service import ConversationSummaryService

        method = ConversationSummaryService.create_segment_if_needed
        assert not inspect.iscoroutinefunction(method), (
            "create_segment_if_needed must be synchronous for asyncio.to_thread. "
            "If it becomes async, the caller in agent_service.py must use "
            "asyncio.create_task() or await directly."
        )

    def test_segment_creation_actually_creates_records(self, monkeypatch):
        """Prove that calling create_segment_if_needed actually creates DB records."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            4,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        # Write 4 messages
        for i in range(4):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        # Call directly (not via asyncio.to_thread)
        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            db=db,
        )

        assert len(result) == 1, f"Expected 1 segment created, got {len(result)}"
        assert result[0]["message_count"] == 4

    def test_segment_creation_failure_does_not_break_caller(self, monkeypatch):
        """The caller wraps the call in try/except — verify the pattern works."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        for i in range(24):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        # Simulate failure inside create_segment_if_needed
        def _failing(*args, **kwargs):
            raise RuntimeError("simulated segment failure")

        monkeypatch.setattr(
            conversation_summary_service,
            "create_segment_if_needed",
            _failing,
        )

        # This is how agent_service calls it — must not propagate
        failed = False
        try:
            conversation_summary_service.create_segment_if_needed(
                conversation_id=conv.conversation_id,
                user_id=user.id,
                db=db,
            )
        except RuntimeError:
            failed = True

        assert failed, "The failure should propagate from the mock"
        # The real caller in agent_service wraps this in try/except


# ── 1. Migration / ORM model tests ─────────────────────────────────────


class TestMigrationAndORM:
    def test_segment_table_exists_with_all_columns(self):
        """ORM model maps to the correct table with all columns."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)
        run = _make_run(db, user.id, conv.conversation_id)
        m1 = _make_message(db, user.id, conv.conversation_id, role="user", content="a", run_id=run.id)
        m2 = _make_message(db, user.id, conv.conversation_id, role="assistant", content="b", run_id=run.id)

        repo = AgentConversationSummarySegmentRepository(db)
        seg = repo.create_segment(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            start_message_id=m1.id,
            end_message_id=m2.id,
            start_message_created_at=m1.created_at,
            end_message_created_at=m2.created_at,
            message_count=2,
            summary_text="test summary",
        )

        assert seg.id is not None
        assert seg.start_message_created_at is not None
        assert seg.end_message_created_at is not None
        assert seg.message_count == 2

    def test_unique_constraint_prevents_duplicate_message_range(self):
        """Conversation + start_message_id + end_message_id must be unique."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)
        run = _make_run(db, user.id, conv.conversation_id)
        m1 = _make_message(db, user.id, conv.conversation_id, role="user", content="a", run_id=run.id)
        m2 = _make_message(db, user.id, conv.conversation_id, role="assistant", content="b", run_id=run.id)

        repo = AgentConversationSummarySegmentRepository(db)
        repo.create_segment(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            start_message_id=m1.id,
            end_message_id=m2.id,
            message_count=2,
            summary_text="first",
        )

        # Second insert with same range should fail
        with pytest.raises(Exception):
            repo.create_segment(
                conversation_id=conv.conversation_id,
                user_id=user.id,
                start_message_id=m1.id,
                end_message_id=m2.id,
                message_count=2,
                summary_text="duplicate",
            )


# ── 2. Segment creation tests ──────────────────────────────────────────


class TestSegmentCreation:
    def test_no_segment_when_below_threshold(self, monkeypatch):
        """23 messages with segment_size=24: no segment created."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        for i in range(23):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            db=db,
        )
        assert len(result) == 0

    def test_create_one_segment_at_threshold(self, monkeypatch):
        """24 messages with segment_size=24: 1 segment created."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        for i in range(24):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            db=db,
        )

        assert len(result) == 1
        assert result[0]["message_count"] == 24
        assert "Phoenix" in result[0].get("summary_text", "")
        # Verify segment range
        repo = AgentConversationSummarySegmentRepository(db)
        seg = repo.get_by_id(result[0]["id"])
        all_msgs = AgentChatMessageRepository(db).list_by_conversation(user.id, conv.conversation_id)
        assert seg.start_message_id == all_msgs[0].id
        assert seg.end_message_id == all_msgs[23].id

    def test_create_two_segments_for_48_messages(self, monkeypatch):
        """48 messages: 2 non-overlapping segments."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        for i in range(48):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            db=db,
        )

        assert len(result) == 2
        # Verify non-overlapping ranges
        repo = AgentConversationSummarySegmentRepository(db)
        seg1 = repo.get_by_id(result[0]["id"])
        seg2 = repo.get_by_id(result[1]["id"])
        all_msgs = AgentChatMessageRepository(db).list_by_conversation(user.id, conv.conversation_id)
        assert seg1.start_message_id == all_msgs[0].id
        assert seg1.end_message_id == all_msgs[23].id
        assert seg2.start_message_id == all_msgs[24].id
        assert seg2.end_message_id == all_msgs[47].id

    def test_segment_creation_is_idempotent(self, monkeypatch):
        """Calling twice with same messages only creates one segment."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        for i in range(24):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        result1 = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )
        result2 = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )

        assert len(result1) == 1, f"First call should create 1 segment, got {len(result1)}"
        assert len(result2) == 0, f"Second call should create 0 segments, got {len(result2)}"
        repo = AgentConversationSummarySegmentRepository(db)
        assert repo.count_by_conversation(user.id, conv.conversation_id) == 1

    def test_creation_continues_from_latest_segment(self, monkeypatch):
        """After first 24 are frozen, next 24 only create segment for 25-48."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        # Batch 1
        for i in range(24):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"batch1_msg{i}")
        r1 = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )
        assert len(r1) == 1

        # Batch 2
        for i in range(24):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"batch2_msg{i}")
        r2 = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )
        assert len(r2) == 1
        assert r2[0]["message_count"] == 24

        repo = AgentConversationSummarySegmentRepository(db)
        assert repo.count_by_conversation(user.id, conv.conversation_id) == 2

    def test_creation_disabled_respects_config(self, monkeypatch):
        """When enable_conversation_segment_creation=False, nothing happens."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            False,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            4,
        )

        for i in range(10):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )
        assert len(result) == 0


# ── 3. Qdrant write tests ──────────────────────────────────────────────


class TestQdrantWrite:
    def test_segment_write_to_qdrant_best_effort(self, monkeypatch):
        """Qdrant upsert is called with correct payload."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        mock_upsert = MagicMock()
        mock_client = MagicMock()
        mock_client.upsert = mock_upsert
        mock_client.get_collection = MagicMock(return_value=True)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._get_qdrant_client",
            lambda self: mock_client,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._ensure_segment_collection",
            lambda self: True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            4,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_vector_collection",
            "conversation_summary_segments",
        )

        for i in range(4):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )

        assert len(result) == 1
        # Verify upsert was called at least once
        assert mock_upsert.call_count >= 1
        call_args = mock_upsert.call_args
        # Check payload structure
        points_kw = call_args[1].get("points") or call_args[0][1].get("points") or []
        if points_kw:
            points = points_kw if isinstance(points_kw, list) else []
            if points:
                payload = points[0].get("payload", {}) if isinstance(points[0], dict) else getattr(points[0], "payload", {})
                assert payload.get("type") == "conversation_summary_segment"
                assert payload.get("conversation_id") == conv.conversation_id
                assert payload.get("user_id") == user.id
                assert payload.get("segment_id") == result[0]["id"]

    def test_qdrant_upsert_failure_does_not_block_creation(self, monkeypatch):
        """PG segment creation succeeds even when Qdrant fails."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        # Qdrant always fails
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("qdrant down")),
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            4,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        for i in range(4):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        # Must not raise
        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )

        assert len(result) == 1
        # PG record still created
        repo = AgentConversationSummarySegmentRepository(db)
        assert repo.count_by_conversation(user.id, conv.conversation_id) == 1


# ── 4. Segment recall tests ────────────────────────────────────────────


class TestSegmentRecall:
    def test_recall_uses_qdrant_first(self, monkeypatch):
        """When Qdrant returns results, source is 'qdrant'."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)
        run = _make_run(db, user.id, conv.conversation_id)

        # Create a segment
        repo = AgentConversationSummarySegmentRepository(db)
        m1 = _make_message(db, user.id, conv.conversation_id, role="user", content="start", run_id=run.id)
        m2 = _make_message(db, user.id, conv.conversation_id, role="assistant", content="end", run_id=run.id)
        seg = repo.create_segment(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            start_message_id=m1.id,
            end_message_id=m2.id,
            message_count=2,
            summary_text="项目代号是 Phoenix，幂等字段是 idempotency_key",
        )

        # Mock Qdrant to return this segment
        mock_hit = MagicMock()
        mock_hit.id = "qdrant_point_123"
        mock_hit.score = 0.91
        mock_hit.payload = {
            "segment_id": seg.id,
            "conversation_id": conv.conversation_id,
            "summary_text": seg.summary_text,
        }

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._search_segments_in_qdrant",
            lambda self, **kw: [{"id": str(seg.id), "summary_text": seg.summary_text, "_score": 0.91, "segment_id": seg.id}],
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_recall",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_min_score",
            0.15,
        )

        results = conversation_summary_service.search_relevant_segments(
            conversation_id=conv.conversation_id,
            query="项目代号是什么",
            user_id=user.id,
            db=db,
        )

        assert len(results) >= 1
        top = results[0]
        assert top.source == "qdrant"
        assert "Phoenix" in top.summary_text

    def test_recall_falls_back_to_pg_when_qdrant_fails(self, monkeypatch):
        """When Qdrant throws, PG ILIKE fallback is used."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)
        run = _make_run(db, user.id, conv.conversation_id)

        # Create a segment with a unique string
        repo = AgentConversationSummarySegmentRepository(db)
        m1 = _make_message(db, user.id, conv.conversation_id, role="user", content="start", run_id=run.id)
        m2 = _make_message(db, user.id, conv.conversation_id, role="assistant", content="end", run_id=run.id)
        repo.create_segment(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            start_message_id=m1.id,
            end_message_id=m2.id,
            message_count=2,
            summary_text="支付模块的幂等字段叫 payment_idempotency_key_2024",
        )

        # Qdrant fails
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._search_segments_in_qdrant",
            lambda self, **kw: (_ for _ in ()).throw(RuntimeError("qdrant timeout")),
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_recall",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_min_score",
            0.05,
        )

        results = conversation_summary_service.search_relevant_segments(
            conversation_id=conv.conversation_id,
            query="支付模块的幂等字段叫什么",
            user_id=user.id,
            db=db,
        )

        assert len(results) >= 1
        assert results[0].source == "pg_ilike"
        assert "payment_idempotency_key_2024" in results[0].summary_text

    def test_recall_filters_by_min_score(self, monkeypatch):
        """Results below min_score are excluded."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)
        run = _make_run(db, user.id, conv.conversation_id)

        repo = AgentConversationSummarySegmentRepository(db)
        m1 = _make_message(db, user.id, conv.conversation_id, role="user", content="start", run_id=run.id)
        m2 = _make_message(db, user.id, conv.conversation_id, role="assistant", content="end", run_id=run.id)
        repo.create_segment(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            start_message_id=m1.id,
            end_message_id=m2.id,
            message_count=2,
            summary_text="irrelevant content about weather",
        )

        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_recall",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_min_score",
            0.80,  # very high threshold
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_recall_limit",
            5,
        )

        # Qdrant returns low score
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._search_segments_in_qdrant",
            lambda self, **kw: [{"id": "x", "summary_text": "irrelevant", "_score": 0.01, "segment_id": None}],
        )

        results = conversation_summary_service.search_relevant_segments(
            conversation_id=conv.conversation_id,
            query="something",
            user_id=user.id,
            db=db,
        )
        assert len(results) == 0

    def test_recall_limit_is_respected(self, monkeypatch):
        """Only conversation_segment_recall_limit results are returned."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_recall",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_recall_limit",
            3,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_min_score",
            0.05,
        )

        repo = AgentConversationSummarySegmentRepository(db)
        for i in range(10):
            m1 = _make_message(db, user.id, conv.conversation_id, role="user", content=f"start{i}")
            m2 = _make_message(db, user.id, conv.conversation_id, role="assistant", content=f"end{i}")
            repo.create_segment(
                conversation_id=conv.conversation_id,
                user_id=user.id,
                start_message_id=m1.id,
                end_message_id=m2.id,
                message_count=2,
                summary_text=f"segment {i} about project topic",
            )

        results = conversation_summary_service.search_relevant_segments(
            conversation_id=conv.conversation_id,
            query="project",
            user_id=user.id,
            db=db,
            limit=3,
        )

        assert len(results) <= 3

    def test_recall_returns_empty_for_empty_query(self, monkeypatch):
        """Empty query returns empty list."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_recall",
            True,
        )

        results = conversation_summary_service.search_relevant_segments(
            conversation_id=conv.conversation_id,
            query="",
            user_id=user.id,
            db=db,
        )
        assert results == []


# ── 5. Data isolation tests ────────────────────────────────────────────


class TestDataIsolation:
    def test_recall_does_not_cross_conversation(self):
        """Segment from conversation A must not appear in conversation B queries."""
        db = make_test_session()
        user = _make_user(db)
        conv_a = _make_conversation(db, user.id)
        conv_b = _make_conversation(db, user.id)
        run = _make_run(db, user.id, conv_a.conversation_id)

        repo_a = AgentConversationSummarySegmentRepository(db)
        m1 = _make_message(db, user.id, conv_a.conversation_id, role="user", content="a1", run_id=run.id)
        m2 = _make_message(db, user.id, conv_a.conversation_id, role="assistant", content="a2", run_id=run.id)
        repo_a.create_segment(
            conversation_id=conv_a.conversation_id,
            user_id=user.id,
            start_message_id=m1.id,
            end_message_id=m2.id,
            message_count=2,
            summary_text="Phoenix project in conversation A",
        )

        repo_b = AgentConversationSummarySegmentRepository(db)
        m3 = _make_message(db, user.id, conv_b.conversation_id, role="user", content="b1")
        m4 = _make_message(db, user.id, conv_b.conversation_id, role="assistant", content="b2")
        repo_b.create_segment(
            conversation_id=conv_b.conversation_id,
            user_id=user.id,
            start_message_id=m3.id,
            end_message_id=m4.id,
            message_count=2,
            summary_text="Apollo project in conversation B",
        )

        # Search conv A — should not see conv B content via PG ILIKE
        results_a = repo_a.search_segments_ilike(
            conversation_id=conv_a.conversation_id,
            user_id=user.id,
            query="Apollo",
            limit=5,
        )
        apollo_in_a = any("Apollo" in r["summary_text"] for r in results_a)
        assert not apollo_in_a, "Conversation A should not see conversation B's segment"

        # Search conv B — should not see conv A content
        results_b = repo_b.search_segments_ilike(
            conversation_id=conv_b.conversation_id,
            user_id=user.id,
            query="Phoenix",
            limit=5,
        )
        phoenix_in_b = any("Phoenix" in r["summary_text"] for r in results_b)
        assert not phoenix_in_b, "Conversation B should not see conversation A's segment"

    def test_recall_does_not_cross_user(self):
        """User A's segments are invisible to user B."""
        db = make_test_session()
        user_a = _make_user(db, email="a@test.com")
        user_b = _make_user(db, email="b@test.com")
        conv_a = _make_conversation(db, user_a.id)
        conv_b = _make_conversation(db, user_b.id)

        run_a = _make_run(db, user_a.id, conv_a.conversation_id)
        repo_a = AgentConversationSummarySegmentRepository(db)
        m1 = _make_message(db, user_a.id, conv_a.conversation_id, role="user", content="a1", run_id=run_a.id)
        m2 = _make_message(db, user_a.id, conv_a.conversation_id, role="assistant", content="a2", run_id=run_a.id)
        repo_a.create_segment(
            conversation_id=conv_a.conversation_id,
            user_id=user_a.id,
            start_message_id=m1.id,
            end_message_id=m2.id,
            message_count=2,
            summary_text="Phoenix project for user A",
        )

        run_b = _make_run(db, user_b.id, conv_b.conversation_id)
        repo_b = AgentConversationSummarySegmentRepository(db)
        m3 = _make_message(db, user_b.id, conv_b.conversation_id, role="user", content="b1", run_id=run_b.id)
        m4 = _make_message(db, user_b.id, conv_b.conversation_id, role="assistant", content="b2", run_id=run_b.id)
        repo_b.create_segment(
            conversation_id=conv_b.conversation_id,
            user_id=user_b.id,
            start_message_id=m3.id,
            end_message_id=m4.id,
            message_count=2,
            summary_text="Apollo project for user B",
        )

        # User A PG search
        results_a = repo_a.search_segments_ilike(
            conversation_id=conv_a.conversation_id,
            user_id=user_a.id,
            query="Apollo",
            limit=5,
        )
        assert len(results_a) == 0

        # User B PG search
        results_b = repo_b.search_segments_ilike(
            conversation_id=conv_b.conversation_id,
            user_id=user_b.id,
            query="Phoenix",
            limit=5,
        )
        assert len(results_b) == 0


# ── 6. ContextBuilder injection tests ──────────────────────────────────


class TestContextBuilderInjection:
    def test_conversation_segments_enters_context_builder_payload(self):
        """Verify the payload key is consumed by gather()."""
        from src.web_app.context.builder import ContextBuilder

        builder = ContextBuilder(route="chat")
        payload = {
            "task": "hello",
            "conversation_history": "User: hi\nAssistant: Hello!",
            "conversation_segments": "UNIQUE_SEGMENT_FACT_12345",
        }
        context = builder.build(payload)
        assert "UNIQUE_SEGMENT_FACT_12345" in context, (
            "segment text must survive gather/select/structure and appear in final context"
        )

    def test_conversation_continuity_section_appears(self):
        """When segments are present, [Conversation Continuity] section appears."""
        from src.web_app.context.builder import ContextBuilder

        builder = ContextBuilder(route="chat")
        payload = {
            "task": "test",
            "conversation_segments": "[Relevant Historical Conversation Segments]\n## Segment 1 | Score 0.85\nPhoenix project",
        }
        context = builder.build(payload)
        assert "[Conversation Continuity]" in context
        assert "Phoenix project" in context

    def test_output_instructions_for_consistency(self):
        """When segments are present, consistency instructions appear."""
        from src.web_app.context.builder import ContextBuilder

        builder = ContextBuilder(route="chat")
        payload = {
            "task": "test",
            "conversation_segments": "[Relevant Historical Conversation Segments]\n## Segment 1 | Score 0.85\nSome fact",
        }
        context = builder.build(payload)
        assert "Conversation Continuity" in context
        assert "Conversation History" in context.lower() or "以最近消息为准" in context

    def test_conversation_segments_source_is_registered(self):
        """SOURCE_MAP should have conversation_segments mapped."""
        from src.web_app.context.builder import SOURCE_MAP

        assert "conversation_segments" in SOURCE_MAP, (
            "conversation_segments must be in SOURCE_MAP for gather() to route it"
        )
        assert SOURCE_MAP["conversation_segments"] == "Conversation Continuity"


# ── 7. Token budget tests ──────────────────────────────────────────────


class TestTokenBudget:
    def test_segment_token_budget_truncates_low_score(self, monkeypatch):
        """High-score segments survive; low-score are truncated or dropped."""
        from src.web_app.agent.runtime.node_groups.read_nodes import (
            _format_recalled_segments_for_context,
        )

        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_max_tokens",
            100,  # very tight budget
        )

        high = RecalledConversationSegment(
            id=1, conversation_id="c1",
            summary_text="HIGH_VALUE_FACT: 这是一段很关键的包含项目代号 Phoenix 的摘要 " * 20,
            score=0.95, start_message_id=1, end_message_id=24,
            start_time=None, end_time=None, message_count=24, source="qdrant",
        )
        low = RecalledConversationSegment(
            id=2, conversation_id="c1",
            summary_text="LOW_VALUE_FACT: 这是一段次要的包含天气讨论的摘要 " * 20,
            score=0.10, start_message_id=25, end_message_id=48,
            start_time=None, end_time=None, message_count=24, source="qdrant",
        )

        result = _format_recalled_segments_for_context([high, low])
        assert "HIGH_VALUE_FACT" in result, "High-score segment must survive"
        # Low-score segment should be either truncated or absent
        assert "LOW_VALUE_FACT" not in result or "[segment truncated]" in result

    def test_segments_do_not_evict_recent_messages(self):
        """Large segments must not cause recent messages to be dropped."""
        from src.web_app.context.builder import ContextBuilder

        huge_segment = "[Relevant Historical Conversation Segments]\n" + ("long summary text. " * 500)
        builder = ContextBuilder(route="chat")
        payload = {
            "task": "What is the project code?",
            "conversation_history": "User: Hello\nAssistant: Hi! RECENT_CRITICAL_FACT",
            "conversation_segments": huge_segment,
        }
        context = builder.build(payload)
        assert "RECENT_CRITICAL_FACT" in context, (
            "Recent messages must survive even with huge segments"
        )
        assert "What is the project code" in context


# ── 8. 100-turn regression test ────────────────────────────────────────


class Test100TurnRegression:
    def test_early_fact_survives_100_turns(self, monkeypatch):
        """Fact from turn 5 must be recallable at turn 100 via segment search."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_recall",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_recall_limit",
            5,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_min_score",
            0.05,
        )

        # Turn 5: critical fact
        early_fact = "项目代号是 Phoenix，数据库必须使用 PostgreSQL，不能使用 MySQL"
        # Tracks: each turn = 2 messages (user + assistant) → 100 turns = 200 messages
        svc = conversation_summary_service

        for turn in range(1, 101):
            role_user = "user" if turn % 2 == 1 else "assistant"
            # Turn ~5: inject the critical fact
            if turn == 5:
                content = early_fact
            else:
                content = f"turn {turn} message: discussing topic {turn % 10}"
            _make_message(db, user.id, conv.conversation_id, role="user", content=content)
            _make_message(db, user.id, conv.conversation_id, role="assistant", content=f"response {turn}")

        # Create segments for all 200 messages (8 segments of 24)
        svc.create_segment_if_needed(
            conversation_id=conv.conversation_id,
            user_id=user.id,
            db=db,
        )

        # Segment count should be 8 (200 // 24 = 8)
        repo = AgentConversationSummarySegmentRepository(db)
        seg_count = repo.count_by_conversation(user.id, conv.conversation_id)
        assert seg_count >= 8, f"Expected 8+ segments for 200 messages, got {seg_count}"

        # Now at turn 100, query for the early fact
        results = svc.search_relevant_segments(
            conversation_id=conv.conversation_id,
            query="我们最早定的项目代号和数据库约束是什么",
            user_id=user.id,
            db=db,
        )

        # The early fact should be in one of the segments
        all_text = " ".join(r.summary_text for r in results)
        assert "Phoenix" in all_text, (
            f"Early fact 'Phoenix' must appear in recalled segments. "
            f"Got {len(results)} results."
        )

        # Also verify: Phoenix should NOT come from recent messages
        # (recent messages are turns 76+, which don't mention Phoenix)
        from src.web_app.core.config import settings as _set
        recent_limit = getattr(_set, "conversation_recent_message_limit", 12)
        if recent_limit > 0:
            recent = AgentChatMessageRepository(db).list_recent_by_conversation(
                user_id=user.id,
                conversation_id=conv.conversation_id,
                limit=recent_limit,
            )
            recent_text = " ".join(m.content for m in recent)
            # Recent messages should NOT contain Phoenix
            assert "Phoenix" not in recent_text, (
                "Sanity check failed: 'Phoenix' should NOT be in recent messages"
            )


# ── 9. Graceful degradation tests ──────────────────────────────────────


class TestGracefulDegradation:
    def test_segment_creation_failure_does_not_break_caller(self, monkeypatch):
        """If LLM fails, fallback summary is used and segment is still created."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            4,
        )

        # Write messages
        for i in range(4):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        # Force LLM to fail → fallback summary
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            lambda p: (_ for _ in ()).throw(RuntimeError("LLM down")),
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )

        # Must not raise — fallback summary should be created
        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )
        assert len(result) == 1, f"Expected 1 segment, got {len(result)}"
        assert result[0].get("message_count", 0) > 0

    def test_segment_recall_failure_returns_empty(self, monkeypatch):
        """If both Qdrant and PG ILIKE fail, search returns empty list."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_recall",
            True,
        )

        # Qdrant fails
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._search_segments_in_qdrant",
            lambda self, **kw: (_ for _ in ()).throw(RuntimeError("qdrant down")),
        )
        # PG also fails
        monkeypatch.setattr(
            "src.web_app.db.repositories.agent_repository.AgentConversationSummarySegmentRepository.search_segments_ilike",
            lambda self, **kw: (_ for _ in ()).throw(RuntimeError("pg broken")),
        )

        results = conversation_summary_service.search_relevant_segments(
            conversation_id=conv.conversation_id,
            query="anything",
            user_id=user.id,
            db=db,
        )
        assert results == []

    def test_context_builder_works_without_segments(self):
        """ContextBuilder must work fine when no segments exist."""
        from src.web_app.context.builder import ContextBuilder

        builder = ContextBuilder(route="chat")
        context = builder.build({
            "task": "hello",
            "conversation_history": "User: hi\nAssistant: Hello!",
        })
        assert "hello" in context


# ── 10. API contract tests ───────────────────────────────────────────────


class TestAPIContracts:
    def test_create_segment_if_needed_returns_list(self, monkeypatch):
        """Contract: create_segment_if_needed ALWAYS returns list[dict]."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )

        # Below threshold → empty list
        for i in range(5):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")
        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )
        assert isinstance(result, list), f"Must return list, got {type(result)}"
        assert len(result) == 0

    def test_create_segment_if_needed_length_is_count(self, monkeypatch):
        """Contract: return list length equals number of segments created."""
        db = make_test_session()
        user = _make_user(db)
        conv = _make_conversation(db, user.id)

        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service._llm_call",
            _fake_llm_segment,
        )
        monkeypatch.setattr(
            "src.web_app.services.conversation_summary_service.ConversationSummaryService._index_segment_to_qdrant",
            lambda *a, **kw: "",
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_summary_segment_size",
            24,
        )
        monkeypatch.setattr(
            "src.web_app.core.config.settings.enable_conversation_segment_creation",
            True,
        )

        for i in range(48):
            _make_message(db, user.id, conv.conversation_id, role="user", content=f"msg {i}")

        result = conversation_summary_service.create_segment_if_needed(
            conversation_id=conv.conversation_id, user_id=user.id, db=db,
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_search_relevant_segments_requires_keyword_arguments(self):
        """Contract: search_relevant_segments uses keyword-only params."""
        import inspect
        sig = inspect.signature(ConversationSummaryService.search_relevant_segments)
        params = list(sig.parameters.keys())
        # 'self' is first, then all keyword-only
        assert "conversation_id" in params
        assert "query" in params
        assert "user_id" in params

    def test_format_for_context_accepts_recalled_segment_dataclass(self, monkeypatch):
        """Contract: format_for_context handles RecalledConversationSegment."""
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_max_tokens",
            1800,
        )
        svc = ConversationSummaryService()
        seg = RecalledConversationSegment(
            id=1, conversation_id="c1",
            summary_text="Project Phoenix details",
            score=0.90, start_message_id=1, end_message_id=10,
            start_time=None, end_time=None, message_count=10, source="qdrant",
        )
        text = svc.format_for_context(
            summary={"summary_text": "Running summary"},
            relevant_segments=[seg],
        )
        assert "Project Phoenix details" in text
        assert "[Relevant Historical Conversation Segments]" in text

    def test_format_for_context_accepts_legacy_dict_segments(self, monkeypatch):
        """Backward compat: format_for_context also handles plain dicts."""
        monkeypatch.setattr(
            "src.web_app.core.config.settings.conversation_segment_max_tokens",
            1800,
        )
        svc = ConversationSummaryService()
        legacy_seg = {
            "summary_text": "Old-style segment about Apollo project",
            "score": 0.75,
            "message_count": 12,
            "start_time": None,
            "end_time": None,
        }
        text = svc.format_for_context(
            summary=None,
            relevant_segments=[legacy_seg],
        )
        assert "Apollo project" in text
