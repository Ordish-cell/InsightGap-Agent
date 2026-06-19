"""Conversation summary service — maintains running summaries and historical segments.

The running summary is updated after each assistant turn, progressively covering
the entire conversation.  Every ~24 messages a frozen segment is created for
future semantic recall.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.agent_repository import (
    AgentChatMessageRepository,
    AgentConversationSummaryRepository,
    AgentConversationSummarySegmentRepository,
)

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────

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

SEGMENT_CREATION_PROMPT = """你是对话历史归档器。给定下面一段对话消息，生成一个不可变的历史摘要段，用于后续语义召回。

要求：
1. 只保留对未来回答有价值的信息。
2. 提取 3-8 个关键词用于语义匹配。
3. 输出 JSON：

{{
  "summary_text": "这段对话的中文摘要, 150 字以内",
  "keywords": ["关键词1", "关键词2"],
  "facts": ["事实1"]
}}

[对话消息]
{messages}

请输出 JSON："""


# ── Helpers ────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output."""
    text = str(text).strip()
    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    # Find the first { ... }
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
    # ChatOpenAI returns an AIMessage with .content
    content = getattr(message, "content", str(message))
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _extract_query_terms(query: str) -> list[str]:
    """Extract searchable tokens from a query string."""
    tokens: list[str] = []
    # CJK characters individually
    cjk = re.findall(r"[一-鿿㐀-䶿]+", query)
    for chunk in cjk:
        tokens.extend(chunk)
    # Alphanumeric tokens (2+ chars)
    alpha = re.findall(r"[a-zA-Z0-9_]{2,}", query)
    tokens.extend(alpha)
    # Deduplicate, keep order, limit
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.append(t)
    return result[:16]


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
        import time as _time
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

        if not settings.enable_conversation_summary:
            return None

        repo = AgentConversationSummaryRepository(db)
        existing = repo.get_by_conversation(user_id, conversation_id)

        # Build prompt inputs
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
            logger.info("conversation_summary.update_after_turn llm_ms=%d", int((time.monotonic() - t_llm) * 1000))
            parsed = _parse_json(raw)
        except Exception as exc:
            logger.exception("conversation_summary.llm_failed error=%s", exc)
            return self.get_summary(conversation_id, user_id, db=db)

        # Extract fields, merge with existing if LLM returns gaps
        summary_text = str(parsed.get("summary_text", ""))[:2000]
        facts = list(parsed.get("facts") or [])[:30]
        preferences = list(parsed.get("preferences") or [])[:20]
        decisions = list(parsed.get("decisions") or [])[:20]
        open_threads = list(parsed.get("open_threads") or [])[:15]
        entities = list(parsed.get("entities") or [])[:30]

        # Determine the latest message_id
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

        # Check if we should create a segment (best-effort, must not crash caller)
        try:
            self.create_segment_if_needed(conversation_id, user_id, db=db)
        except Exception as exc:
            logger.warning("conversation_summary.segment_creation_call_failed error=%s", exc)

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

    def create_segment_if_needed(
        self,
        conversation_id: str,
        user_id: int,
        *,
        db: Session | None = None,
    ) -> dict[str, Any] | None:
        """Create a frozen historical segment if enough new messages have accumulated."""
        if db is None:
            return None
        from src.web_app.core.config import settings

        if not settings.enable_conversation_summary:
            return None

        segment_repo = AgentConversationSummarySegmentRepository(db)
        msg_repo = AgentChatMessageRepository(db)
        summary_repo = AgentConversationSummaryRepository(db)

        existing_summary = summary_repo.get_by_conversation(user_id, conversation_id)
        if existing_summary is None:
            return None

        segment_size = settings.conversation_summary_segment_size
        total_covered = existing_summary.covered_message_count or 0
        segment_count = segment_repo.count_by_conversation(user_id, conversation_id)

        # Create segment when remaining uncovered messages >= segment_size
        if total_covered - segment_count * segment_size < segment_size:
            return None

        all_messages = msg_repo.list_by_conversation(user_id, conversation_id)
        start_idx = segment_count * segment_size
        end_idx = min(start_idx + segment_size, len(all_messages))
        segment_messages = all_messages[start_idx:end_idx]

        if not segment_messages:
            return None

        formatted: list[dict[str, str]] = []
        for m in segment_messages:
            role = getattr(m, "role", "unknown")
            content = (getattr(m, "content", "") or "")[:600]
            formatted.append({"role": role, "content": content})

        prompt = SEGMENT_CREATION_PROMPT.format(
            messages=json.dumps(formatted, ensure_ascii=False),
        )

        try:
            raw = _llm_call(prompt)
            parsed = _parse_json(raw)
        except Exception as exc:
            logger.exception("conversation_summary.segment_llm_failed error=%s", exc)
            return None

        segment_text = str(parsed.get("summary_text", ""))[:1500]
        keywords = list(parsed.get("keywords") or [])[:10]
        facts = list(parsed.get("facts") or [])[:10]

        start_msg_id = getattr(segment_messages[0], "id", None)
        end_msg_id = getattr(segment_messages[-1], "id", None)

        try:
            segment_repo.create(
                user_id=user_id,
                conversation_id=conversation_id,
                start_message_id=start_msg_id,
                end_message_id=end_msg_id,
                summary_text=segment_text,
                keywords_json=keywords,
                facts_json=facts,
            )
        except Exception as exc:
            logger.exception("conversation_summary.segment_db_write_failed error=%s", exc)

        # Optionally index into Qdrant
        self._index_segment_to_qdrant(
            conversation_id=conversation_id,
            user_id=user_id,
            segment_text=segment_text,
            keywords=keywords,
            facts=facts,
        )

        return {
            "start_message_id": start_msg_id,
            "end_message_id": end_msg_id,
            "summary_text": segment_text,
            "keywords": keywords,
            "facts": facts,
        }

    def search_relevant_segments(
        self,
        conversation_id: str,
        user_id: int,
        query: str,
        *,
        top_k: int = 5,
        db: Session | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant historical segments matching the query.

        Prefers Qdrant vector search, falls back to PostgreSQL keyword matching.
        """
        if db is None:
            return []
        from src.web_app.core.config import settings

        if not settings.enable_conversation_segment_recall:
            return []

        # Try Qdrant first
        t0 = time.monotonic()
        qdrant_results = self._search_segments_in_qdrant(
            conversation_id=conversation_id,
            user_id=user_id,
            query=query,
            top_k=top_k,
        )
        t_qdrant = time.monotonic() - t0
        if qdrant_results:
            logger.info("conversation_summary.search_segments backend=qdrant qdrant_ms=%d results=%d", int(t_qdrant * 1000), len(qdrant_results))
            return qdrant_results[:top_k]

        # Fallback to PostgreSQL keyword search
        t1 = time.monotonic()
        segment_repo = AgentConversationSummarySegmentRepository(db)
        terms = _extract_query_terms(query)
        rows = segment_repo.search_by_keywords(
            user_id=user_id,
            conversation_id=conversation_id,
            query_terms=terms,
            limit=top_k,
        )
        logger.info("conversation_summary.search_segments backend=postgres qdrant_ms=%d pg_ms=%d query_terms=%d results=%d",
                    int(t_qdrant * 1000), int((time.monotonic() - t1) * 1000), len(terms), len(rows))

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({
                "id": row.id,
                "summary_text": row.summary_text,
                "keywords": list(row.keywords_json or []),
                "facts": list(row.facts_json or []),
                "start_message_id": row.start_message_id,
                "end_message_id": row.end_message_id,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            })
        return results

    def format_for_context(
        self,
        summary: dict[str, Any] | None = None,
        relevant_segments: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format summary and segments into a structured text block for LLM injection.

        The output uses XML-style markers so the LLM can clearly distinguish
        different memory layers.
        """
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
            for i, seg in enumerate(relevant_segments, 1):
                seg_text = str(seg.get("summary_text", "") or "").strip()
                if seg_text:
                    segment_lines.append(f"{i}. {seg_text}")
            if segment_lines:
                parts.append("[Relevant Older History]\n" + "\n".join(segment_lines))

        parts.append("</conversation_memory>")
        return "\n\n".join(parts)

    # ── Qdrant helpers ─────────────────────────────────────────────

    def _index_segment_to_qdrant(
        self,
        conversation_id: str,
        user_id: int,
        segment_text: str,
        keywords: list[str],
        facts: list[str],
    ) -> None:
        """Index a segment into Qdrant for semantic search."""
        try:
            from src.web_app.core.config import settings
            from src.web_app.rag.embeddings import embed_text
            import uuid

            client = self._get_qdrant_client()
            if client is None:
                return

            collection_name = getattr(settings, "conversation_segments_qdrant_collection", "conversation_segments")
            vector = embed_text(segment_text)
            point_id = str(uuid.uuid4())
            client.upsert(
                collection_name=collection_name,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "summary_text": segment_text,
                        "keywords": keywords,
                        "facts": facts,
                    },
                }],
            )
        except Exception as exc:
            logger.warning("conversation_summary.qdrant_index_failed error=%s", exc)

    def _search_segments_in_qdrant(
        self,
        conversation_id: str,
        user_id: int,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search segments in Qdrant by semantic similarity."""
        try:
            from src.web_app.core.config import settings
            from src.web_app.rag.embeddings import embed_text
            from qdrant_client.models import Filter, MatchValue

            client = self._get_qdrant_client()
            if client is None:
                return []

            collection_name = getattr(settings, "conversation_segments_qdrant_collection", "conversation_segments")

            # Ensure collection exists
            t_check = time.monotonic()
            try:
                client.get_collection(collection_name)
            except Exception:
                logger.info("conversation_summary.qdrant_search collection_missing=%s check_ms=%d",
                           collection_name, int((time.monotonic() - t_check) * 1000))
                return []

            t_embed = time.monotonic()
            query_vector = embed_text(query)
            logger.info("conversation_summary.qdrant_search embed_ms=%d query_len=%d",
                       int((time.monotonic() - t_embed) * 1000), len(query))

            t_search = time.monotonic()
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=Filter(must=[
                    {"key": "conversation_id", "match": MatchValue(value=conversation_id)},
                    {"key": "user_id", "match": MatchValue(value=user_id)},
                ]),
                limit=top_k,
                score_threshold=0.25,
            )

            logger.info("conversation_summary.qdrant_search search_ms=%d embed_ms=%d results=%d",
                       int((time.monotonic() - t_search) * 1000),
                       int((time.monotonic() - t_embed) * 1000),
                       len(results))

            return [
                {
                    "id": hit.id,
                    "summary_text": hit.payload.get("summary_text", ""),
                    "keywords": hit.payload.get("keywords", []),
                    "facts": hit.payload.get("facts", []),
                    "_score": hit.score,
                }
                for hit in results
            ]
        except Exception as exc:
            logger.warning("conversation_summary.qdrant_search_failed error=%s", exc)
            return []


# ── Singleton ──────────────────────────────────────────────────────────

conversation_summary_service = ConversationSummaryService()
