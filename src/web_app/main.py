from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.web_app.api.v1.router import api_router
from src.web_app.core.config import settings

app = FastAPI(title="Open Deep Research Agent OS API")

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
    import logging
    _log = logging.getLogger(__name__)

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
        _log.info(
            "[STARTUP] checkpointer health: healthy=%s backend=%s durable=%s tables=%s",
            health["healthy"], health["backend"], health["durable"],
            health["tables_present"],
        )
    except RuntimeError as exc:
        _log.critical("[STARTUP] checkpointer health check FAILED: %s", exc)
        raise
    except Exception as exc:
        _log.error("[STARTUP] checkpointer health check error (non-fatal): %s", exc)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "FastAPI is running",
    }


app.include_router(api_router, prefix="/api/v1")
