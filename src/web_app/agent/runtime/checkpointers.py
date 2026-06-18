"""LangGraph checkpointer factory — supports postgres, redis, memory.

Includes startup health check for production hardening (Phase 13).
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

# ── Health check ────────────────────────────────────────────────────────

CHECKPOINT_TABLES = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
)


def check_checkpointer_health(
    *,
    backend: str = "postgres",
    conn_string: str = "",
    require_durable: bool = False,
    checkpointer_enabled: bool = True,
    db_session_factory=None,
) -> dict[str, Any]:
    """Verify the configured checkpointer backend is healthy at startup.

    Returns a health dict with keys: backend, durable, healthy, tables_present,
    saver_type, error (if any).

    Raises RuntimeError if require_durable=True and the check fails.
    """
    result: dict[str, Any] = {
        "backend": backend,
        "durable": False,
        "healthy": False,
        "tables_present": [],
        "saver_type": None,
        "error": None,
    }

    if not checkpointer_enabled:
        _log.info("[CHECKPOINTER_HEALTH] checkpointer disabled — skipping health check")
        result["healthy"] = True
        result["note"] = "checkpointer disabled"
        return result

    backend = backend.lower().strip()

    if backend == "memory":
        result["durable"] = False
        if require_durable:
            msg = (
                "[CHECKPOINTER_HEALTH] FATAL: backend=memory but "
                "require_durable=True. Set AGENT_CHECKPOINTER_BACKEND=postgres "
                "for production."
            )
            _log.error(msg)
            result["error"] = msg
            raise RuntimeError(msg)
        _log.warning(
            "[CHECKPOINTER_HEALTH] backend=memory — checkpoints will be "
            "lost on process restart.  This is acceptable for dev/test only."
        )
        result["healthy"] = True
        result["note"] = "memory backend — dev/test only"
        return result

    if backend == "redis":
        result["durable"] = True
        _log.warning(
            "[CHECKPOINTER_HEALTH] backend=redis is experimental. "
            "langgraph-checkpoint-redis==0.4.1 has a known bug with "
            "Command(resume=...).  Use postgres for production."
        )
        result["note"] = "redis backend — experimental (known Command(resume) bug)"
        result["healthy"] = True
        return result

    if backend == "postgres":
        result["durable"] = True
        if not conn_string:
            msg = (
                "[CHECKPOINTER_HEALTH] FATAL: backend=postgres requires "
                "a connection string but none was provided."
            )
            _log.error(msg)
            result["error"] = msg
            if require_durable:
                raise RuntimeError(msg)
            return result

        # 1. Verify PostgresSaver can be created.
        try:
            saver = _PostgresSaverHandle.create(conn_string)
            result["saver_type"] = type(saver).__name__
            _log.info(
                "[CHECKPOINTER_HEALTH] PostgresSaver created successfully "
                "type=%s", result["saver_type"],
            )
        except Exception as exc:
            msg = (
                f"[CHECKPOINTER_HEALTH] FATAL: PostgresSaver creation failed: {exc}"
            )
            _log.error(msg, exc_info=True)
            result["error"] = msg
            if require_durable:
                raise RuntimeError(msg) from exc
            return result

        # 2. Verify the 4 checkpoint tables exist.
        missing = _verify_checkpoint_tables(conn_string)
        result["tables_present"] = [
            t for t in CHECKPOINT_TABLES if t not in missing
        ]
        if missing:
            msg = (
                f"[CHECKPOINTER_HEALTH] FATAL: missing checkpoint tables: {missing}. "
                "Run PostgresSaver.setup() to create them."
            )
            _log.error(msg)
            result["error"] = msg
            if require_durable:
                raise RuntimeError(msg)
            return result

        _log.info(
            "[CHECKPOINTER_HEALTH] all %d checkpoint tables present: %s",
            len(CHECKPOINT_TABLES), ", ".join(CHECKPOINT_TABLES),
        )
        result["healthy"] = True
        return result

    # Unknown backend
    msg = f"[CHECKPOINTER_HEALTH] unknown backend={backend}"
    _log.error(msg)
    result["error"] = msg
    if require_durable:
        raise RuntimeError(msg)
    return result


def _verify_checkpoint_tables(conn_string: str) -> list[str]:
    """Check which checkpoint tables are missing from PostgreSQL.

    Returns a list of missing table names (empty = all present).
    """
    try:
        import psycopg
    except ImportError:
        _log.warning(
            "[CHECKPOINTER_HEALTH] psycopg not installed — "
            "cannot verify checkpoint tables"
        )
        return []

    missing: list[str] = []
    try:
        conn = psycopg.connect(conn_string)
        try:
            cur = conn.cursor()
            for table in CHECKPOINT_TABLES:
                try:
                    cur.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables "
                        "WHERE table_name = %s)",
                        (table,),
                    )
                    row = cur.fetchone()
                    if not row or not row[0]:
                        missing.append(table)
                except Exception:
                    missing.append(table)
        finally:
            conn.close()
    except Exception as exc:
        _log.warning(
            "[CHECKPOINTER_HEALTH] cannot connect to verify tables: %s", exc
        )
        return list(CHECKPOINT_TABLES)

    return missing


class _PostgresSaverHandle:
    """Holds a sync PostgresSaver and its connection-pool context manager alive.

    Used for sync invoke() only (tests, dev).  Production uses
    AsyncPostgresSaver via _AsyncPostgresSaverHandle.
    """

    @staticmethod
    def create(conn_string: str) -> Any:
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore

        ctx = PostgresSaver.from_conn_string(conn_string)
        saver: Any = ctx.__enter__()
        try:
            saver.setup()
        except Exception:
            pass
        saver._checkpointer_ctx = ctx  # type: ignore[attr-defined]
        return saver


class _AsyncPostgresSaverHandle:
    """Holds an AsyncPostgresSaver and its async connection-pool context alive.

    AsyncPostgresSaver implements aget_tuple, aput, etc. — required for
    graph.ainvoke() which the AgentRuntime uses in production.
    """

    @staticmethod
    async def create(conn_string: str) -> Any:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore

        ctx = AsyncPostgresSaver.from_conn_string(conn_string)
        saver: Any = await ctx.__aenter__()
        try:
            await saver.setup()
        except Exception:
            pass
        saver._checkpointer_ctx = ctx  # type: ignore[attr-defined]
        return saver


def build_checkpointer(
    *,
    backend: str = "postgres",
    conn_string: str = "",
    redis_url: str | None = None,
    redis_password: str = "",
    redis_key_prefix: str = "langgraph:checkpoint:",
    require_durable: bool = False,
) -> Any:
    """LangGraph checkpointer factory.

    Args:
        backend: "postgres" | "redis" | "memory"
        conn_string: PostgreSQL connection string (for postgres).
        redis_url: Redis URL (for redis).
        redis_password: Redis password.
        redis_key_prefix: Prefix for checkpoint keys.
        require_durable: If True, fail fast when only memory is available.

    Returns:
        A checkpointer instance, or None if memory is unavailable
        and require_durable is False.

    Raises:
        RuntimeError: If require_durable=True and the requested backend
            cannot provide durable checkpoint storage.
    """
    backend = backend.lower().strip()

    # ── postgres ────────────────────────────────────────────────────
    if backend == "postgres":
        if not conn_string:
            if require_durable:
                raise RuntimeError(
                    "[CHECKPOINTER] backend=postgres requires a connection string"
                )
            _log.warning(
                "[CHECKPOINTER] postgres conn_string is empty — "
                "falling back to memory"
            )
            return _build_memory_saver(require_durable)
        try:
            saver = _PostgresSaverHandle.create(conn_string)
            _log.info(
                "[CHECKPOINTER] backend=postgres saver_type=PostgresSaver "
                "durable=True"
            )
            return saver
        except Exception as exc:
            if require_durable:
                raise RuntimeError(
                    f"[CHECKPOINTER] backend=postgres unavailable: {exc}"
                ) from exc
            _log.warning(
                "[CHECKPOINTER] PostgresSaver unavailable — "
                "falling back to memory. error=%s", exc
            )
            return _build_memory_saver(require_durable)

    # ── redis ───────────────────────────────────────────────────────
    if backend == "redis":
        if not redis_url:
            if require_durable:
                raise RuntimeError(
                    "[CHECKPOINTER] backend=redis requires redis_url"
                )
            _log.warning(
                "[CHECKPOINTER] redis_url is empty — falling back to memory"
            )
            return _build_memory_saver(require_durable)
        try:
            from langgraph.checkpoint.redis import RedisSaver  # type: ignore

            conn_kwargs: dict[str, Any] = {}
            if redis_password:
                conn_kwargs["password"] = redis_password

            saver = RedisSaver(
                redis_url,
                connection_args=conn_kwargs if conn_kwargs else None,
                checkpoint_prefix=redis_key_prefix,
                checkpoint_write_prefix=redis_key_prefix + "write",
            )
            try:
                saver.setup()
            except Exception as setup_exc:
                _log.warning(
                    "[CHECKPOINTER] RedisSaver.setup() failed — "
                    "indexes may need manual creation. error=%s", setup_exc
                )
            _log.info(
                "[CHECKPOINTER] backend=redis saver_type=RedisSaver "
                "durable=True url=%s", redis_url
            )
            return saver
        except Exception as exc:
            if require_durable:
                raise RuntimeError(
                    f"[CHECKPOINTER] backend=redis unavailable: {exc}"
                ) from exc
            _log.warning(
                "[CHECKPOINTER] RedisSaver unavailable — "
                "falling back to memory. error=%s", exc
            )
            return _build_memory_saver(require_durable)

    # ── memory ──────────────────────────────────────────────────────
    if backend == "memory":
        return _build_memory_saver(require_durable)

    # ── unknown backend ─────────────────────────────────────────────
    if require_durable:
        raise RuntimeError(
            f"[CHECKPOINTER] unknown backend={backend} and require_durable=True"
        )
    _log.warning(
        "[CHECKPOINTER] unknown backend=%s — falling back to memory", backend
    )
    return _build_memory_saver(False)


def _build_memory_saver(require_durable: bool) -> Any:
    if require_durable:
        raise RuntimeError(
            "[CHECKPOINTER] require_durable=True but only memory is available. "
            "Set AGENT_CHECKPOINTER_BACKEND=postgres for durable checkpoints."
        )
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        _log.warning(
            "[CHECKPOINTER] backend=memory saver_type=InMemorySaver durable=False — "
            "checkpoint state will be lost on process restart"
        )
        return InMemorySaver()
    except Exception as exc:
        _log.warning(
            "[CHECKPOINTER] InMemorySaver unavailable: %s", exc
        )
        return None
