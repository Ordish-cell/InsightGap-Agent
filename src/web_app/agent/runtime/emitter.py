"""Unified RuntimeEventEmitter — single source of truth for all SSE events.

Every runtime node uses this emitter to produce events with consistent:
- run_id, conversation_id, message_id, event_seq, display_channel, created_at
- DB persistence via record_event
- SSE queue push via queue_stream_event
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.agent.runtime.events import DISPLAY_CHANNEL_THINKING, DISPLAY_CHANNEL_ANSWER, DISPLAY_CHANNEL_TOOL, DISPLAY_CHANNEL_STATUS, queue_stream_event

logger = logging.getLogger(__name__)


class RuntimeEventEmitter:
    """Centralised event emitter used by all runtime nodes.

    Each run gets one emitter.  The emitter auto-increments event_seq and
    attaches run_id/conversation_id/message_id/created_at to every event.

    Usage inside a node::

        emitter = RuntimeEventEmitter(db, state)
        await emitter.thought("正在分析请求…")
        await emitter.tool("tool_call_started", {"tool_name": "email.send"})
        await emitter.status("approval_required", {"approval_id": "..."})
    """

    def __init__(self, db: Session, state: dict[str, Any]):
        self.db = db
        self._state = state
        self._seq = state.setdefault("_event_seq", 0)

    # ── helpers ──────────────────────────────────────────────────

    @property
    def run_id(self) -> int | None:
        return self._state.get("run_id")

    @property
    def thread_id(self) -> str:
        return self._state.get("thread_id", "")

    @property
    def user_id(self) -> int | None:
        return self._state.get("user_id")

    @property
    def conversation_id(self) -> str:
        return self._state.get("conversation_id", "")

    @property
    def message_id(self) -> str:
        return self._state.get("message_id", "")

    def _next_seq(self) -> int:
        self._seq += 1
        self._state["_event_seq"] = self._seq
        return self._seq

    def _queue(self) -> Any:
        return self._state.get("_stream_queue")

    # ── public emit ──────────────────────────────────────────────

    async def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        display_channel: str = DISPLAY_CHANNEL_STATUS,
        persist: bool = True,
        node_name: str = "",
    ) -> None:
        """Emit a single event to SSE queue and optionally persist to DB."""
        seq = self._next_seq()
        run_id = self.run_id
        thread_id = self.thread_id
        user_id = self.user_id
        now = datetime.now(UTC).replace(tzinfo=None).isoformat()

        full_payload = {
            "run_id": run_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "event_seq": seq,
            "display_channel": display_channel,
            "created_at": now,
            **payload,
        }

        queue = self._queue()
        if queue:
            queue_stream_event(
                queue,
                event_type,
                full_payload,
                run_id=run_id,
                thread_id=thread_id,
                node_name=node_name,
            )
            await asyncio.sleep(0)

        if persist and self.db and run_id:
            try:
                record_event(
                    self.db,
                    run_id,
                    event_type,
                    full_payload,
                    node_name=node_name,
                    user_id=user_id,
                    thread_id=thread_id,
                )
            except Exception:
                logger.exception("emit: persist failed event_type=%s run_id=%s", event_type, run_id)

    async def thought(self, text: str, *, node_name: str = "", status: str = "streaming") -> None:
        """Emit a visible_thought_delta line."""
        await self.emit(
            "visible_thought_delta",
            {"text": text, "status": status, "source": "visible_thought"},
            display_channel=DISPLAY_CHANNEL_THINKING,
            node_name=node_name or "runtime",
        )

    async def status(self, event_type: str, payload: dict[str, Any] | None = None, *, node_name: str = "") -> None:
        """Emit a status event (run_paused, run_resumed, approval_required, etc.)."""
        await self.emit(
            event_type,
            payload or {},
            display_channel=DISPLAY_CHANNEL_STATUS,
            node_name=node_name or "runtime",
        )

    async def tool(self, event_type: str, payload: dict[str, Any], *, node_name: str = "") -> None:
        """Emit a tool event (tool_call_started/completed/failed)."""
        await self.emit(
            event_type,
            payload,
            display_channel=DISPLAY_CHANNEL_TOOL,
            node_name=node_name or "tool_agent",
        )

    async def answer_delta(self, text: str, *, index: int = 0, node_name: str = "") -> None:
        """Emit an answer_delta chunk."""
        await self.emit(
            "answer_delta",
            {"text": text, "index": index},
            display_channel=DISPLAY_CHANNEL_ANSWER,
            node_name=node_name or "final_response",
        )

    async def answer_started(self, *, node_name: str = "") -> None:
        await self.emit(
            "answer_started",
            {},
            display_channel=DISPLAY_CHANNEL_ANSWER,
            node_name=node_name or "final_response",
        )

    async def answer_completed(self, answer: str, *, node_name: str = "") -> None:
        await self.emit(
            "answer_completed",
            {"answer": answer},
            display_channel=DISPLAY_CHANNEL_ANSWER,
            node_name=node_name or "final_response",
        )

    async def run_created(
        self,
        *,
        user_message: dict[str, Any] | None = None,
        assistant_message: dict[str, Any] | None = None,
        conversation: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            "run_created",
            {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "conversation": conversation,
            },
            display_channel=DISPLAY_CHANNEL_STATUS,
        )

    async def run_completed(self, answer: str, *, status: str = "completed") -> None:
        await self.emit(
            "run_completed",
            {"status": status, "answer": answer},
            display_channel=DISPLAY_CHANNEL_STATUS,
        )

    async def run_failed(self, error: str, *, answer: str = "") -> None:
        await self.emit(
            "run_failed",
            {"status": "failed", "error": error, "answer": answer},
            display_channel=DISPLAY_CHANNEL_STATUS,
        )

    async def run_paused(self, approval_id: Any = None, *, reason: str = "approval_required") -> None:
        await self.emit(
            "run_paused",
            {"status": "waiting_approval", "approval_id": approval_id, "reason": reason},
            display_channel=DISPLAY_CHANNEL_STATUS,
        )

    async def run_resumed(self) -> None:
        await self.emit(
            "run_resumed",
            {"status": "resuming"},
            display_channel=DISPLAY_CHANNEL_STATUS,
        )

    async def approval_required(
        self,
        *,
        approval_id: Any,
        tool_name: str,
        risk_level: str,
        preview: dict[str, Any],
        safety_notes: list[str] | None = None,
        actions: list[str] | None = None,
    ) -> None:
        """Emit approval_required with a frontend-ready payload."""
        await self.emit(
            "approval_required",
            {
                "approval_id": approval_id,
                "tool_name": tool_name,
                "risk_level": risk_level,
                "title": f"需要你确认：{_tool_display_name(tool_name)}",
                "preview": preview,
                "safety_notes": safety_notes or [],
                "actions": actions or ["approve", "reject"],
                "status": "pending",
            },
            display_channel=DISPLAY_CHANNEL_STATUS,
            node_name="tool_agent",
        )

    async def approval_granted(self, approval_id: Any) -> None:
        await self.emit(
            "approval_granted",
            {"approval_id": approval_id, "status": "approved"},
            display_channel=DISPLAY_CHANNEL_STATUS,
        )

    async def approval_rejected(self, approval_id: Any) -> None:
        await self.emit(
            "approval_rejected",
            {"approval_id": approval_id, "status": "rejected"},
            display_channel=DISPLAY_CHANNEL_STATUS,
        )


def _tool_display_name(tool_name: str) -> str:
    names = {
        "email.send": "发送邮件",
        "local_file.write": "写入文件",
        "local_file.append": "追加文件",
        "local_file.delete": "删除文件",
        "local_file.read": "读取文件",
        "local_file.list": "列出文件",
        "web.search": "联网搜索",
        "system.time": "读取本地时间",
        "system.calc": "本地计算",
        "system.unit_convert": "本地单位换算",
        "system.uuid": "生成 UUID",
        "system.hash": "本地哈希计算",
    }
    return names.get(tool_name, tool_name)
