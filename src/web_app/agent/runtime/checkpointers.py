from typing import Any


def build_checkpointer(redis_url: str | None = None) -> Any:
    """Best-effort LangGraph checkpointer.

    Redis checkpointing is optional in local/dev because RedisJSON or
    RediSearch may be unavailable. The runtime falls back to MemorySaver.
    """
    if redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver  # type: ignore

            return RedisSaver.from_conn_string(redis_url)
        except Exception:
            pass
    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except Exception:
        return None
