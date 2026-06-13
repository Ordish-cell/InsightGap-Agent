import logging
from typing import Any


_log = logging.getLogger(__name__)


def build_checkpointer(redis_url: str | None = None) -> Any:
    """Best-effort LangGraph checkpointer.

    Redis checkpointing is optional in local/dev because RedisJSON or
    RediSearch may be unavailable. The runtime falls back to MemorySaver.
    """
    if redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver  # type: ignore

            return RedisSaver.from_conn_string(redis_url)
        except Exception as exc:
            _log.warning("Redis checkpointer unavailable; falling back to MemorySaver: %s", exc)
    else:
        _log.warning("Redis checkpointer URL is empty; falling back to MemorySaver")
    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except Exception as exc:
        _log.warning("MemorySaver checkpointer unavailable; running without LangGraph checkpointing: %s", exc)
        return None
