import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.web_app.api.v1.router import api_router
from src.web_app.core.config import settings

app = FastAPI(title="Open Deep Research Agent OS API")

_log_main = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", "")
    return await call_next(request)


@app.on_event("startup")
async def startup_health_checks():
    """Verify production dependencies before accepting traffic."""
    # ── Checkpointer health ──────────────────────────────────────
    try:
        from src.web_app.agent.runtime.checkpointers import check_checkpointer_health

        backend = getattr(settings, "agent_checkpointer_backend", "postgres")
        require_durable = getattr(settings, "agent_checkpointer_require_durable", True)
        cp_enabled = getattr(settings, "agent_langgraph_checkpointer_enabled", True)
        cp_db_url = (
            getattr(settings, "agent_checkpointer_database_url", "")
            or getattr(settings, "database_url", "").replace("+psycopg2", "")
        )

        health = check_checkpointer_health(
            backend=backend,
            conn_string=cp_db_url,
            require_durable=require_durable,
            checkpointer_enabled=cp_enabled,
        )
        _log_main.info(
            "[STARTUP] checkpointer health: healthy=%s backend=%s durable=%s tables=%s",
            health["healthy"], health["backend"], health["durable"],
            health["tables_present"],
        )
    except RuntimeError as exc:
        _log_main.critical("[STARTUP] checkpointer health check FAILED: %s", exc)
        raise
    except Exception as exc:
        _log_main.error("[STARTUP] checkpointer health check error (non-fatal): %s", exc)

    # ── Background checkpoint / approval cleanup ──────────────────
    _launch_cleanup_scheduler()


@app.on_event("shutdown")
async def shutdown_agent_runs():
    from src.web_app.services.agent_run_task_manager import agent_run_task_manager

    await agent_run_task_manager.shutdown()


def _launch_cleanup_scheduler() -> None:
    """Start a background asyncio task for periodic checkpoint / approval cleanup."""
    enabled = getattr(settings, "agent_checkpoint_cleanup_enabled", True)
    interval = getattr(settings, "agent_checkpoint_cleanup_interval_minutes", 60)
    if not enabled or interval <= 0:
        _log_main.info(
            "[CLEANUP_SCHEDULER] disabled (enabled=%s interval=%s)", enabled, interval
        )
        return

    _log_main.info(
        "[CLEANUP_SCHEDULER] starting background cleanup every %d minutes", interval
    )
    asyncio.create_task(_cleanup_loop(interval))


async def _cleanup_loop(interval_minutes: int) -> None:
    """Run checkpoint/approval cleanup on a fixed interval."""
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            from src.web_app.agent.runtime.checkpoint_cleanup import (
                cleanup_checkpoints,
                cleanup_orphan_checkpoints,
            )
            from src.web_app.services.approval_expiry import expire_stale_approvals

            # 1. Expire stale approvals first (so runs move to "expired" status)
            approval_summary = expire_stale_approvals()
            _log_main.info(
                "[CLEANUP_SCHEDULER] approval expiry: expired=%s",
                approval_summary.get("expired_count", 0),
            )

            # 2. TTL-based checkpoint cleanup
            cp_summary = cleanup_checkpoints()
            _log_main.info(
                "[CLEANUP_SCHEDULER] checkpoint TTL cleanup: "
                "deleted_threads=%s deleted_rows=%s",
                len(cp_summary.get("deleted_threads", [])),
                cp_summary.get("deleted_rows", 0),
            )

            # 3. Orphan checkpoint cleanup (safety net)
            orphan_summary = cleanup_orphan_checkpoints()
            _log_main.info(
                "[CLEANUP_SCHEDULER] orphan cleanup: "
                "orphan_thread_ids=%s deleted_rows=%s",
                len(orphan_summary.get("orphan_thread_ids", [])),
                orphan_summary.get("deleted_rows", 0),
            )
        except Exception:
            _log_main.exception(
                "[CLEANUP_SCHEDULER] cleanup cycle failed — will retry in %d minutes",
                interval_minutes,
            )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "FastAPI is running",
    }


app.include_router(api_router, prefix="/api/v1")
