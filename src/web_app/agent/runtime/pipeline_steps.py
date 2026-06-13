from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


PIPELINE_TITLES: dict[str, str] = {
    "understand_request": "理解问题",
    "load_conversation_context": "加载会话上下文",
    "load_memory_context": "加载长期记忆",
    "memory_writer": "处理记忆写入",
    "generate_answer": "生成回答",
    "finalize": "完成",
}


def append_pipeline_step(
    state: dict[str, Any],
    key: str,
    *,
    status: str = "completed",
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append or replace a user-facing pipeline step.

    This is intentionally separate from visible_thoughts/trace_events:
    pipeline_steps are progress labels for the UI, not model reasoning.
    """
    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    steps = [item for item in list(state.get("pipeline_steps") or []) if item.get("key") != key]
    item: dict[str, Any] = {
        "key": key,
        "title": PIPELINE_TITLES.get(key, key),
        "status": status,
        "detail": detail,
        "started_at": now,
        "completed_at": now if status in {"completed", "failed", "skipped"} else None,
    }
    if extra:
        item.update(extra)
    steps.append(item)
    state["pipeline_steps"] = steps
    return state


def ensure_pipeline_step(
    state: dict[str, Any],
    key: str,
    *,
    status: str = "completed",
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if any(item.get("key") == key for item in list(state.get("pipeline_steps") or [])):
        return state
    return append_pipeline_step(state, key, status=status, detail=detail, extra=extra)
