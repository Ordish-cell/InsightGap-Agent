"""Conversation summary service — maintains running summaries and historical segments.

The running summary is updated after each assistant turn, progressively covering
the entire conversation.  Every ~N messages a frozen segment is created for
future semantic recall via Qdrant vector search (fallback to PG ILIKE).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.agent_repository import (
    AgentChatMessageRepository,
    AgentConversationSummaryRepository,
    AgentConversationSummarySegmentRepository,
)

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────


@dataclass
class RecalledConversationSegment:
    id: int
    conversation_id: str
    summary_text: str
    score: float
    start_message_id: int | None
    end_message_id: int | None
    start_time: Any
    end_time: Any
    message_count: int
    source: str  # "qdrant" | "pg_ilike"


# ── Prompts ────────────────────────────────────────────────────────────

CONVERSATION_SUMMARY_UPDATE_PROMPT = """你是对话记忆压缩器。你的任务是合并已有 running summary 和新一轮消息，生成更新后的长期对话摘要。

要求：
1. 保留对未来回答有帮助的信息。
2. 删除寒暄、重复、无意义过程。
3. 不要编造任何未在消息中出现的内容。
4. 如果新消息推翻旧信息，使用新信息。
5. 明确保留：用户偏好、项目背景、未完成任务、关键实体（文件名、表名、函数名）、代码路径、数据库模式、重要决策、错误结论。
6. 输出 JSON，字段如下（都是数组, 如果没有则返回空数组）：

{{
  "summary_text": "一段 200 字以内的中文摘要，覆盖最近对话的主要内容",
  "facts": ["事实1", "事实2"],
  "preferences": ["偏好1"],
  "decisions": ["决定1"],
  "open_threads": ["未完成事项1"],
  "entities": ["实体/文件名/表名/函数名1"]
}}

已知信息：
[现有摘要]
{existing_summary}

[新消息]
{new_messages}

请输出 JSON："""

SEGMENT_CREATION_PROMPT = """你是对话历史压缩器。请把以下对话片段压缩成可供未来检索和恢复上下文的历史段摘要。

要求：
1. 保留后续可能有用的事实、决定、约束、用户偏好、项目状态、任务进度。
2. 保留具体名词、文件名、函数名、配置名、bug 现象、方案选择。
3. 删除寒暄、重复确认、无用过程。
4. 不要编造。
5. 如果片段里出现冲突信息，以片段中更晚的消息为准。
6. 输出结构必须稳定，使用以下 Markdown 格式：

## Segment Scope
- Conversation ID:
- Message count:
- Time range:

## Key Facts
- ...

## Decisions
- ...

## Technical Details
- ...

## Open Issues
- ...

## Follow-up Tasks
- ...

【对话片段】
{messages}

请输出历史段摘要："""


# ── Helpers ────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output."""
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("conversation_summary.parse_json_failed raw_preview=%s", text[:200])
        return {}


def _llm_call(prompt: str) -> str:
    """Synchronous LLM call using the memory tier model."""
    from src.web_app.agent.llm.factory import get_chat_model

    model = get_chat_model("memory", complexity="low", temperature=0.15)
    message = model.invoke(prompt)
    content = getattr(message, "content", str(message))
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _count_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for mixed CJK/ASCII."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Service ────────────────────────────────────────────────────────────


class ConversationSummaryService:
    """Maintains running summaries and historical segments for conversations."""

    def __init__(self):
        self._qdrant_client = None
        self._qdrant_init_attempted = False

    def _get_qdrant_client(self):
        """Lazy-init QdrantClient, reused across calls. Returns None if unavailable."""
        if self._qdrant_client is not None:
            return self._qdrant_client
        if self._qdrant_init_attempted:
            return None
        self._qdrant_init_attempted = True
        try:
            from src.web_app.core.config import settings
            from qdrant_client import QdrantClient

            if not settings.qdrant_url:
                return None
            kwargs = {"url": settings.qdrant_url, "timeout": settings.qdrant_timeout}
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            self._qdrant_client = QdrantClient(**kwargs)
            return self._qdrant_client
        except Exception as exc:
            logger.warning("conversation_summary.qdrant_client_init_failed error=%s", exc)
            return None

    def get_summary(
        self, conversation_id: str, user_id: int, *, db: Session | None = None
    ) -> dict[str, Any] | None:
        """Return the current running summary for a conversation."""
        if db is None:
            return None
        repo = AgentConversationSummaryRepository(db)
        row = repo.get_by_conversation(user_id, conversation_id)
        if row is None:
            return None
        return {
            "id": row.id,
            "summary_text": row.summary_text,
            "facts": list(row.facts_json or []),
            "decisions": list(row.decisions_json or []),
            "open_threads": list(row.open_threads_json or []),
            "preferences": list(row.preferences_json or []),
            "entities": list(row.entities_json or []),
            "last_message_id": row.last_message_id,
            "covered_message_count": row.covered_message_count,
            "summary_version": row.summary_version,
        }

    def update_after_turn(
        self,
        conversation_id: str,
        user_id: int,
        new_messages: list[dict[str, str]],
        *,
        db: Session | None = None,
    ) -> dict[str, Any] | None:
        """Update the running summary after one or more new messages.

        Called after assistant final_response is written to agent_chat_messages.
        """
        if db is None:
            return None
        from src.web_app.core.config import settings

        if not getattr(settings, "enable_conversation_summary", False):
            return None

        repo = AgentConversationSummaryRepository(db)
        existing = repo.get_by_conversation(user_id, conversation_id)

        existing_text = json.dumps({
            "summary_text": existing.summary_text if existing else "",
            "facts": list(existing.facts_json or []) if existing else [],
            "preferences": list(existing.preferences_json or []) if existing else [],
            "decisions": list(existing.decisions_json or []) if existing else [],
            "open_threads": list(existing.open_threads_json or []) if existing else [],
            "entities": list(existing.entities_json or []) if existing else [],
        }, ensure_ascii=False)

        new_text = json.dumps(new_messages, ensure_ascii=False)

        prompt = CONVERSATION_SUMMARY_UPDATE_PROMPT.format(
            existing_summary=existing_text,
            new_messages=new_text,
        )

        try:
            t_llm = time.monotonic()
            raw = _llm_call(prompt)
            logger.info(
                "conversation_summary.update_after_turn llm_ms=%d",
                int((time.monotonic() - t_llm) * 1000),
            )
            parsed = _parse_json(raw)
        except Exception as exc:
            logger.exception("conversation_summary.llm_failed error=%s", exc)
            return self.get_summary(conversation_id, user_id, db=db)

        summary_text = str(parsed.get("summary_text", ""))[:2000]
        facts = list(parsed.get("facts") or [])[:30]
        preferences = list(parsed.get("preferences") or [])[:20]
        decisions = list(parsed.get("decisions") or [])[:20]
        open_threads = list(parsed.get("open_threads") or [])[:15]
        entities = list(parsed.get("entities") or [])[:30]

        last_message_id = existing.last_message_id if existing else None
        new_count = (existing.covered_message_count if existing else 0) + len(new_messages)

        values = {
            "summary_text": summary_text,
            "facts_json": facts,
            "decisions_json": decisions,
            "open_threads_json": open_threads,
            "preferences_json": preferences,
            "entities_json": entities,
            "last_message_id": last_message_id,
            "covered_message_count": new_count,
        }
        if existing:
            values["summary_version"] = existing.summary_version + 1

        try:
            repo.upsert(user_id, conversation_id, **values)
        except Exception as exc:
            logger.exception("conversation_summary.db_write_failed error=%s", exc)

        return {
            "summary_text": summary_text,
            "facts": facts,
            "decisions": decisions,
            "open_threads": open_threads,
            "preferences": preferences,
            "entities": entities,
            "last_message_id": last_message_id,
            "covered_message_count": new_count,
        }

    # ── Segment creation ─────────────────────────────────────────────

    def create_segment_if_needed(
        self,
        *,
        conversation_id: str,
        user_id: int,
        db: Session | None = None,
    ) -> list[dict[str, Any]]:
        """Create frozen historical segments when enough new messages accumulate.

        When db is None (e.g. called from agent_service background thread),
        creates its own session. Callers must catch exceptions.
        """
        _own_db = False
        if db is None:
            from src.web_app.db.session import SessionLocal
            db = SessionLocal()
            _own_db = True

        try:
            return self._create_segment_if_needed_impl(
                conversation_id=conversation_id,
                user_id=user_id,
                db=db,
            )
        finally:
            if _own_db:
                db.close()

    def _create_segment_if_needed_impl(
        self,
        *,
        conversation_id: str,
        user_id: int,
        db: Session,
    ) -> list[dict[str, Any]]:

        from src.web_app.core.config import settings

        if not getattr(settings, "enable_conversation_segment_creation", True):
            logger.debug(
                "conversation_summary.segment_creation_disabled",
                extra={"conversation_id": conversation_id},
            )
            return []

        segment_size = getattr(settings, "conversation_summary_segment_size", 24)

        segment_repo = AgentConversationSummarySegmentRepository(db)
        msg_repo = AgentChatMessageRepository(db)

        # 1. Find the latest segment to know where we left off
        latest_segment = segment_repo.get_latest_segment(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        # 2. Fetch all messages after the latest frozen segment
        pending_messages = segment_repo.list_messages_after_segment(
            conversation_id=conversation_id,
            user_id=user_id,
            after_message_id=latest_segment.end_message_id if latest_segment else None,
        )

        if len(pending_messages) < segment_size:
            logger.debug(
                "conversation_summary.segment_not_needed",
                extra={
                    "conversation_id": conversation_id,
                    "pending_count": len(pending_messages),
                    "segment_size": segment_size,
                },
            )
            return []

        # 3. Create segments for each full batch
        created: list[dict[str, Any]] = []
        idx = 0
        while idx + segment_size <= len(pending_messages):
            batch = pending_messages[idx : idx + segment_size]
            first_msg = batch[0]
            last_msg = batch[-1]

            start_id = getattr(first_msg, "id", None)
            end_id = getattr(last_msg, "id", None)

            # Guard against re-creating the same segment
            if start_id is not None and end_id is not None:
                if segment_repo.segment_exists(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    start_message_id=start_id,
                    end_message_id=end_id,
                ):
                    idx += segment_size
                    continue

            # 4. Build LLM prompt from batch messages
            formatted = _format_messages_for_segment(batch)
            prompt = SEGMENT_CREATION_PROMPT.format(
                messages=json.dumps(formatted, ensure_ascii=False),
            )

            summary_text = ""
            keywords: list[str] = []
            facts: list[str] = []
            t_llm = 0.0

            try:
                t_start = time.monotonic()
                raw = _llm_call(prompt)
                t_llm = int((time.monotonic() - t_start) * 1000)
                summary_text = str(raw).strip()[:3000]
            except Exception as exc:
                logger.exception(
                    "conversation_summary.segment_llm_failed",
                    extra={
                        "conversation_id": conversation_id,
                        "start_message_id": start_id,
                        "end_message_id": end_id,
                        "error": str(exc),
                    },
                )
                summary_text = _fallback_segment_summary(batch)

            # 5. Persist to PostgreSQL
            start_ts = getattr(first_msg, "created_at", None)
            end_ts = getattr(last_msg, "created_at", None)

            try:
                segment = segment_repo.create_segment(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    start_message_id=start_id,
                    end_message_id=end_id,
                    start_message_created_at=start_ts,
                    end_message_created_at=end_ts,
                    message_count=len(batch),
                    summary_text=summary_text,
                    keywords_json=keywords,
                    facts_json=facts,
                    embedding_id="",
                )
            except Exception as exc:
                logger.exception(
                    "conversation_summary.segment_db_write_failed",
                    extra={
                        "conversation_id": conversation_id,
                        "start_message_id": start_id,
                        "end_message_id": end_id,
                        "error": str(exc),
                    },
                )
                idx += segment_size
                continue

            # 6. Try indexing into Qdrant (best-effort, must not fail)
            qdrant_id = ""
            try:
                qdrant_id = self._index_segment_to_qdrant(
                    segment_id=segment.id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    summary_text=summary_text,
                    start_message_id=start_id,
                    end_message_id=end_id,
                    message_count=len(batch),
                )
                if qdrant_id:
                    segment_repo.update_embedding_id(
                        segment_id=segment.id,
                        embedding_id=qdrant_id,
                    )
            except Exception as exc:
                logger.warning(
                    "conversation_summary.segment_qdrant_index_failed",
                    extra={
                        "conversation_id": conversation_id,
                        "segment_id": segment.id,
                        "error": str(exc),
                    },
                )

            logger.info(
                "conversation_summary.segment_created",
                extra={
                    "conversation_id": conversation_id,
                    "segment_id": segment.id,
                    "start_message_id": start_id,
                    "end_message_id": end_id,
                    "message_count": len(batch),
                    "llm_ms": t_llm,
                    "qdrant_id": qdrant_id,
                },
            )

            created.append({
                "id": segment.id,
                "conversation_id": conversation_id,
                "start_message_id": start_id,
                "end_message_id": end_id,
                "message_count": len(batch),
                "summary_text": summary_text[:200],
                "qdrant_id": qdrant_id,
            })

            idx += segment_size

        return created

    # ── Segment recall ───────────────────────────────────────────────

    def search_relevant_segments(
        self,
        *,
        conversation_id: str,
        query: str,
        user_id: int,
        db: Session | None = None,
        limit: int | None = None,
    ) -> list[RecalledConversationSegment]:
        """Search for relevant historical segments matching the query.

        Prefers Qdrant vector search, falls back to PostgreSQL ILIKE.
        """
        if db is None:
            return []

        from src.web_app.core.config import settings

        if not getattr(settings, "enable_conversation_segment_recall", True):
            logger.debug(
                "conversation_summary.segment_recall_disabled",
                extra={"conversation_id": conversation_id},
            )
            return []

        if not query or not query.strip():
            return []

        effective_limit = limit or getattr(settings, "conversation_segment_recall_limit", 5)
        min_score = getattr(settings, "conversation_segment_min_score", 0.15)

        results: list[RecalledConversationSegment] = []
        source = "pg_ilike"
        qdrant_failed = False

        # Try Qdrant first
        try:
            qdrant_results = self._search_segments_in_qdrant(
                conversation_id=conversation_id,
                user_id=user_id,
                query=query,
                top_k=effective_limit,
            )
            if qdrant_results:
                source = "qdrant"
                segment_repo = AgentConversationSummarySegmentRepository(db)
                for hit in qdrant_results:
                    # segment_id (from Qdrant payload) is the PG segment id
                    seg_id = hit.get("segment_id") or hit.get("id")
                    if seg_id is None:
                        continue
                    try:
                        db_id = int(seg_id)
                    except (ValueError, TypeError):
                        continue
                    seg = segment_repo.get_by_id(db_id)
                    if seg is None or seg.conversation_id != conversation_id:
                        continue
                    score = float(hit.get("_score", 0))
                    if score < min_score:
                        continue
                    results.append(RecalledConversationSegment(
                        id=seg.id,
                        conversation_id=seg.conversation_id,
                        summary_text=seg.summary_text,
                        score=score,
                        start_message_id=seg.start_message_id,
                        end_message_id=seg.end_message_id,
                        start_time=seg.start_message_created_at,
                        end_time=seg.end_message_created_at,
                        message_count=seg.message_count or 0,
                        source="qdrant",
                    ))
        except Exception as exc:
            logger.warning(
                "conversation_summary.qdrant_search_failed",
                extra={
                    "conversation_id": conversation_id,
                    "error": str(exc),
                },
            )
            qdrant_failed = True

        # Fallback to PostgreSQL ILIKE
        if qdrant_failed or not results:
            try:
                segment_repo = AgentConversationSummarySegmentRepository(db)
                pg_hits = segment_repo.search_segments_ilike(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    query=query,
                    limit=effective_limit,
                )
                for hit in pg_hits:
                    score = float(hit.get("score", 0))
                    if score < min_score:
                        continue
                    results.append(RecalledConversationSegment(
                        id=hit["id"],
                        conversation_id=hit["conversation_id"],
                        summary_text=hit["summary_text"],
                        score=score,
                        start_message_id=hit.get("start_message_id"),
                        end_message_id=hit.get("end_message_id"),
                        start_time=hit.get("start_time"),
                        end_time=hit.get("end_time"),
                        message_count=hit.get("message_count", 0),
                        source="pg_ilike",
                    ))
                if pg_hits:
                    source = "pg_ilike"
            except Exception as exc:
                logger.exception(
                    "conversation_summary.pg_ilike_failed",
                    extra={
                        "conversation_id": conversation_id,
                        "error": str(exc),
                    },
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "conversation_summary.segment_recall",
            extra={
                "conversation_id": conversation_id,
                "segment_recall_source": source,
                "segment_recall_fallback_used": qdrant_failed,
                "recalled_segments": len(results),
                "selected_segment_ids": [r.id for r in results[:effective_limit]],
                "query_preview": query[:100],
            },
        )

        return results[:effective_limit]

    def format_for_context(
        self,
        summary: dict[str, Any] | None = None,
        relevant_segments: list[RecalledConversationSegment] | None = None,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Format summary and segments into a structured text block for LLM injection."""

        from src.web_app.core.config import settings
        effective_max = max_tokens or getattr(settings, "conversation_segment_max_tokens", 1800)

        if not summary and not relevant_segments:
            return ""

        parts: list[str] = ["<conversation_memory>"]

        if summary:
            text = str(summary.get("summary_text", "") or "").strip()
            if text:
                parts.append("[Running Summary]\n" + text)

            facts = list(summary.get("facts") or [])
            if facts:
                parts.append("[Stable Facts]\n" + "\n".join(f"- {f}" for f in facts))

            prefs = list(summary.get("preferences") or [])
            if prefs:
                parts.append("[User Preferences]\n" + "\n".join(f"- {p}" for p in prefs))

            decs = list(summary.get("decisions") or [])
            if decs:
                parts.append("[Decisions]\n" + "\n".join(f"- {d}" for d in decs))

            threads = list(summary.get("open_threads") or [])
            if threads:
                parts.append("[Open Threads / Unresolved Tasks]\n" + "\n".join(f"- {t}" for t in threads))

            ents = list(summary.get("entities") or [])
            if ents:
                parts.append("[Key Entities]\n" + "\n".join(f"- {e}" for e in ents))

        if relevant_segments:
            segment_lines: list[str] = []
            token_budget = effective_max
            for i, seg in enumerate(relevant_segments, 1):
                # Backward compat: accept both dataclass and dict
                if isinstance(seg, dict):
                    seg_text = str(seg.get("summary_text", "") or "").strip()
                    seg_score = seg.get("score", 0)
                    seg_count = seg.get("message_count", 0)
                    seg_start = seg.get("start_time")
                    seg_end = seg.get("end_time")
                else:
                    seg_text = str(getattr(seg, "summary_text", "") or "").strip()
                    seg_score = getattr(seg, "score", 0)
                    seg_count = getattr(seg, "message_count", 0)
                    seg_start = getattr(seg, "start_time", None)
                    seg_end = getattr(seg, "end_time", None)
                if not seg_text:
                    continue
                header = (
                    f"### Segment {i}  |  Score {seg_score:.2f}"
                    f"  |  messages {seg_count}"
                )
                if seg_start and seg_end:
                    st = _fmt_ts(seg_start)
                    et = _fmt_ts(seg_end)
                    header += f"  |  {st} ~ {et}"
                block = header + "\n" + seg_text
                block_tokens = _count_tokens(block)
                if block_tokens > token_budget:
                    # Truncate segment text to fit remaining budget
                    if token_budget > _count_tokens(header) + 10:
                        available = max(0, token_budget - _count_tokens(header) - _count_tokens("\n[segment truncated]"))
                        seg_text = seg_text[:available * 4] + "\n[segment truncated]"
                        block = header + "\n" + seg_text
                    else:
                        continue
                segment_lines.append(block)
                token_budget -= _count_tokens(block)
            if segment_lines:
                parts.append(
                    "[Relevant Historical Conversation Segments]\n"
                    + "\n\n".join(segment_lines)
                )

        # ── Consistency instruction ─────────────────────────────────
        parts.append(
            "[Output Instructions]\n"
            "回答时必须优先保持与 Conversation Continuity 和 Recent Conversation 的一致性。"
            "如果历史摘要、历史段、最近消息之间存在冲突，以最近消息为准；"
            "如果最近消息没有覆盖，则以历史段和 running summary 为准。"
            "不要声称记得没有出现在上下文中的信息。"
        )

        parts.append("</conversation_memory>")
        return "\n\n".join(parts)

    # ── Qdrant helpers ─────────────────────────────────────────────

    def _ensure_segment_collection(self) -> bool:
        """Ensure the Qdrant segment collection exists. Returns True if ready."""
        client = self._get_qdrant_client()
        if client is None:
            return False
        try:
            from src.web_app.core.config import settings
            from qdrant_client.models import Distance, VectorParams

            collection_name = getattr(settings, "conversation_segment_vector_collection", "conversation_summary_segments")
            try:
                client.get_collection(collection_name)
            except Exception:
                from src.web_app.core.config import settings as app_settings
                vector_size = getattr(app_settings, "qdrant_vector_size", 384)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(
                    "conversation_summary.qdrant_collection_created",
                    extra={"collection": collection_name, "vector_size": vector_size},
                )
            return True
        except Exception as exc:
            logger.warning("conversation_summary.qdrant_collection_setup_failed error=%s", exc)
            return False

    def _index_segment_to_qdrant(
        self,
        segment_id: int,
        conversation_id: str,
        user_id: int,
        summary_text: str,
        start_message_id: int | None,
        end_message_id: int | None,
        message_count: int,
    ) -> str:
        """Index a segment into Qdrant for semantic search. Returns point id or empty string."""
        from src.web_app.core.config import settings
        from src.web_app.rag.embeddings import embed_text

        client = self._get_qdrant_client()
        if client is None:
            return ""

        if not self._ensure_segment_collection():
            return ""

        collection_name = getattr(settings, "conversation_segment_vector_collection", "conversation_summary_segments")
        vector = embed_text(summary_text)
        point_id = str(uuid.uuid4())

        client.upsert(
            collection_name=collection_name,
            points=[{
                "id": point_id,
                "vector": vector,
                "payload": {
                    "type": "conversation_summary_segment",
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "segment_id": segment_id,
                    "start_message_id": start_message_id,
                    "end_message_id": end_message_id,
                    "message_count": message_count,
                },
            }],
        )
        return point_id

    def _search_segments_in_qdrant(
        self,
        conversation_id: str,
        user_id: int,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search segments in Qdrant by semantic similarity."""
        from src.web_app.core.config import settings
        from src.web_app.rag.embeddings import embed_text
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_qdrant_client()
        if client is None:
            return []

        collection_name = getattr(settings, "conversation_segment_vector_collection", "conversation_summary_segments")

        # Ensure collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            return []

        t_embed = time.monotonic()
        query_vector = embed_text(query)
        logger.debug(
            "conversation_summary.qdrant_search embed_ms=%d query_len=%d",
            int((time.monotonic() - t_embed) * 1000),
            len(query),
        )

        t_search = time.monotonic()
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=[
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=conversation_id),
                ),
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                ),
            ]),
            limit=top_k,
            score_threshold=0.25,
        )

        logger.debug(
            "conversation_summary.qdrant_search search_ms=%d embed_ms=%d results=%d",
            int((time.monotonic() - t_search) * 1000),
            int((time.monotonic() - t_embed) * 1000),
            len(results),
        )

        return [
            {
                "id": hit.id,
                "summary_text": hit.payload.get("summary_text", ""),
                "_score": hit.score,
                "segment_id": hit.payload.get("segment_id"),
            }
            for hit in results
        ]


# ── Helpers ────────────────────────────────────────────────────────────


def _format_messages_for_segment(messages: list[Any]) -> list[dict[str, str]]:
    """Format AgentChatMessage rows into role/content pairs for segment LLM prompt."""
    formatted: list[dict[str, str]] = []
    for m in messages:
        role = getattr(m, "role", "unknown")
        content = (getattr(m, "content", "") or "")[:800]
        formatted.append({"role": role, "content": content})
    return formatted


def _fallback_segment_summary(messages: list[Any]) -> str:
    """Generate a basic summary when LLM is unavailable."""
    lines: list[str] = ["## Segment Scope"]
    if messages:
        first = messages[0]
        last = messages[-1]
        lines.append(f"- Message count: {len(messages)}")
        lines.append(f"- Time range: {_fmt_ts(getattr(first, 'created_at', None))} ~ {_fmt_ts(getattr(last, 'created_at', None))}")
        lines.append("")
        lines.append("## Key Facts")
        for m in messages[:8]:
            role = getattr(m, "role", "")
            content = (getattr(m, "content", "") or "")[:120]
            if content:
                lines.append(f"- [{role}] {content}")
    return "\n".join(lines)


def _fmt_ts(ts: Any) -> str:
    """Format a datetime into ISO string, handle None."""
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, datetime):
            return ts.isoformat()[:19]
    except Exception:
        pass
    return str(ts)[:19]


def _extract_query_terms(query: str) -> list[str]:
    """Extract searchable tokens from a query string.

    Public for backward compat — used by test_conversation_memory.py.
    """
    tokens: list[str] = []
    cjk = re.findall(r"[一-鿿㐀-䶿]+", query)
    for chunk in cjk:
        tokens.extend(chunk)
    alpha = re.findall(r"[a-zA-Z0-9_]{2,}", query)
    tokens.extend(alpha)
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.append(t)
    return result[:16]


# ── Singleton ──────────────────────────────────────────────────────────

conversation_summary_service = ConversationSummaryService()
