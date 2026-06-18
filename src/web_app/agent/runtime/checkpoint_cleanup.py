"""Checkpoint cleanup — remove stale checkpoint data for finished runs.

Checkpoints are stored in 4 PostgresSaver-managed tables.  Without cleanup,
they grow unbounded.  This module provides a safe cleanup function that:

- Only deletes checkpoints for completed / failed / cancelled runs
- Never touches waiting_approval (paused) runs
- Uses TTLs so recent runs are kept for debugging
- Deletes by thread_id (= "run:{run_id}") across all 3 data tables

Default TTLs:
    completed  7 days
    failed    30 days
    cancelled  7 days

Usage (cron / admin):
    from src.web_app.agent.runtime.checkpoint_cleanup import cleanup_checkpoints
    summary = cleanup_checkpoints()
    print(summary)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

_log = logging.getLogger(__name__)

# ── TTL configuration ──────────────────────────────────────────────────

DEFAULT_TTL_DAYS: dict[str, int] = {
    "completed": 7,
    "failed": 30,
    "cancelled": 7,
    "expired": 7,
}

# Statuses that are safe to clean up (the run is fully done).
CLEANABLE_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})

# These must NEVER be cleaned.
PROTECTED_STATUSES = frozenset({"waiting_approval", "paused", "resuming", "running"})


def _pg_conn_string() -> str:
    from src.web_app.core.config import settings

    raw = (
        getattr(settings, "agent_checkpointer_database_url", "")
        or getattr(settings, "database_url", "")
    )
    return raw.replace("+psycopg2", "")


def cleanup_checkpoints(
    *,
    ttl_days: dict[str, int] | None = None,
    dry_run: bool = False,
    db=None,
) -> dict:
    """Delete checkpoint rows for finished runs older than their TTL.

    Args:
        ttl_days: Override default TTLs, e.g. {"completed": 3, "failed": 14}.
        dry_run: If True, only report what would be deleted.
        db: Optional SQLAlchemy session (for test injection).

    Returns:
        Summary dict with keys:
            deleted_threads, deleted_rows, errors, dry_run, scanned_runs,
            protected_runs, per_status
    """
    ttl = dict(DEFAULT_TTL_DAYS)
    if ttl_days:
        ttl.update(ttl_days)

    summary: dict = {
        "dry_run": dry_run,
        "scanned_runs": 0,
        "protected_runs": 0,
        "deleted_threads": [],
        "deleted_rows": 0,
        "errors": [],
        "per_status": {},
    }

    # ── 1. Find eligible runs from agent_runs ──────────────────────
    eligible_thread_ids: dict[str, list[int]] = {}  # thread_id -> [run_id, ...]

    try:
        own_db = db is None
        if own_db:
            from src.web_app.db.session import SessionLocal
            db = SessionLocal()
        try:
            from src.web_app.models.orm import AgentRun as _AgentRun
            from sqlalchemy import select as _select

            now = datetime.now(timezone.utc)

            # Scan each cleanable status
            from sqlalchemy import func as _sa_func

            for status in sorted(CLEANABLE_STATUSES):
                cutoff = now - timedelta(days=ttl.get(status, 30))
                cutoff_naive = cutoff.replace(tzinfo=None)

                # Use COALESCE so runs with null completed_at fall back
                # to created_at — they still age out instead of leaking forever.
                stmt = (
                    _select(_AgentRun)
                    .where(
                        _AgentRun.status == status,
                        _sa_func.coalesce(
                            _AgentRun.completed_at, _AgentRun.created_at
                        )
                        < cutoff_naive,
                    )
                )
                rows = list(db.execute(stmt).scalars())
                summary["scanned_runs"] += len(rows)

                run_ids_for_status: list[int] = []
                for run in rows:
                    tid = f"run:{run.id}"
                    eligible_thread_ids.setdefault(tid, []).append(run.id)
                    run_ids_for_status.append(run.id)

                summary["per_status"][status] = {
                    "candidates": len(rows),
                    "run_ids": run_ids_for_status,
                }

            # ── 2. Verify no protected runs are in the set ─────────
            # (belt-and-suspenders: re-query to ensure we don't
            #  accidentally delete a run that was paused)
            for status in PROTECTED_STATUSES:
                stmt = _select(_AgentRun).where(_AgentRun.status == status)
                protected = list(db.execute(stmt).scalars())
                for run in protected:
                    tid = f"run:{run.id}"
                    if tid in eligible_thread_ids:
                        _log.error(
                            "[CHECKPOINT_CLEANUP] SAFETY: run %s has "
                            "status=%s but was candidate for deletion — "
                            "removing from eligible set",
                            run.id, run.status,
                        )
                        del eligible_thread_ids[tid]
                        summary["protected_runs"] += 1
                        summary.setdefault("safety_catches", []).append(
                            {"run_id": run.id, "status": run.status}
                        )
        finally:
            if own_db:
                db.close()
    except Exception as exc:
        _log.exception("[CHECKPOINT_CLEANUP] failed to scan agent_runs")
        summary["errors"].append(f"scan_error: {exc}")
        return summary

    if not eligible_thread_ids:
        _log.info("[CHECKPOINT_CLEANUP] no eligible checkpoints to clean")
        return summary

    # ── 3. Delete from checkpoint tables ──────────────────────────
    conn_string = _pg_conn_string()
    if not conn_string:
        summary["errors"].append("no database connection string")
        return summary

    try:
        import psycopg

        conn = psycopg.connect(conn_string)
        try:
            cur = conn.cursor()
            for tid in sorted(eligible_thread_ids.keys()):
                run_ids = eligible_thread_ids[tid]
                _log.info(
                    "[CHECKPOINT_CLEANUP] %s thread_id=%s run_ids=%s",
                    "DRY_RUN would delete" if dry_run else "deleting",
                    tid, run_ids,
                )

                if dry_run:
                    # Count rows without deleting
                    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                        cur.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE thread_id = %s",
                            (tid,),
                        )
                        row = cur.fetchone()
                        if row:
                            summary["deleted_rows"] += row[0]
                    summary["deleted_threads"].append(
                        {"thread_id": tid, "run_ids": run_ids}
                    )
                else:
                    rows_deleted = 0
                    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                        cur.execute(
                            f"DELETE FROM {table} WHERE thread_id = %s",
                            (tid,),
                        )
                        rows_deleted += cur.rowcount
                    summary["deleted_rows"] += rows_deleted
                    summary["deleted_threads"].append(
                        {"thread_id": tid, "run_ids": run_ids, "rows": rows_deleted}
                    )
                    _log.info(
                        "[CHECKPOINT_CLEANUP] deleted %d rows for thread_id=%s "
                        "(run_ids=%s)", rows_deleted, tid, run_ids,
                    )

            if not dry_run:
                conn.commit()
                _log.info(
                    "[CHECKPOINT_CLEANUP] committed — %d total rows deleted "
                    "across %d thread_ids",
                    summary["deleted_rows"], len(eligible_thread_ids),
                )
        finally:
            conn.close()
    except Exception as exc:
        _log.exception("[CHECKPOINT_CLEANUP] delete failed")
        summary["errors"].append(f"delete_error: {exc}")

    return summary


def estimate_checkpoint_size() -> dict:
    """Return rough size estimate of checkpoint data (for monitoring)."""
    conn_string = _pg_conn_string()
    if not conn_string:
        return {"error": "no database connection string"}

    result: dict = {"tables": {}, "total_rows": 0}
    try:
        import psycopg

        conn = psycopg.connect(conn_string)
        try:
            cur = conn.cursor()
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                row = cur.fetchone()
                count = row[0] if row else 0
                result["tables"][table] = count
                result["total_rows"] += count

            # Rough size estimate
            cur.execute(
                "SELECT pg_size_pretty(pg_total_relation_size(%s))",
                ("checkpoints",),
            )
            row = cur.fetchone()
            result["checkpoints_size"] = row[0] if row else "unknown"
        finally:
            conn.close()
    except Exception as exc:
        result["error"] = str(exc)

    return result
