"""Checkpoint cleanup — remove stale checkpoint data for finished runs.

Checkpoints are stored in 4 PostgresSaver-managed tables.  Without cleanup,
they grow unbounded.  This module provides:

- TTL-based cleanup for completed/failed/cancelled/expired runs
- Cascading delete when a conversation is hard-deleted
- Orphan checkpoint cleanup for runs whose agent_runs row is gone
- Size estimation for monitoring

Never touches waiting_approval / paused / resuming / running runs.

Default TTLs:
    completed  7 days
    failed    30 days
    cancelled  7 days
    expired    7 days

Usage:
    from src.web_app.agent.runtime.checkpoint_cleanup import (
        cleanup_checkpoints,
        delete_checkpoints_for_runs,
        cleanup_orphan_checkpoints,
        estimate_checkpoint_size,
    )
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


# ── Cascading delete helpers ─────────────────────────────────────────────

CHECKPOINT_DATA_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


def _thread_ids_from_run_ids(run_ids: list[int]) -> list[str]:
    return [f"run:{rid}" for rid in run_ids]


def delete_checkpoints_for_runs(run_ids: list[int]) -> int:
    """Delete all checkpoint data for the given run_ids.

    Deletes from checkpoints, checkpoint_blobs, and checkpoint_writes.
    Does NOT touch checkpoint_migrations (schema ledger).

    Args:
        run_ids: List of agent_runs.id values to clean up.

    Returns:
        Total number of rows deleted across all 3 tables.
    """
    if not run_ids:
        return 0

    thread_ids = _thread_ids_from_run_ids(run_ids)
    conn_string = _pg_conn_string()
    if not conn_string:
        _log.warning(
            "[CHECKPOINT_CLEANUP] delete_checkpoints_for_runs: "
            "no database connection string — skipping %d run_ids",
            len(run_ids),
        )
        return 0

    total = 0
    try:
        import psycopg

        conn = psycopg.connect(conn_string)
        try:
            cur = conn.cursor()
            for table in CHECKPOINT_DATA_TABLES:
                cur.execute(
                    f"DELETE FROM {table} WHERE thread_id = ANY(%s)",
                    (thread_ids,),
                )
                total += cur.rowcount
            conn.commit()
            _log.info(
                "[CHECKPOINT_CLEANUP] delete_checkpoints_for_runs: "
                "deleted %d rows across 3 tables for %d run_ids "
                "(thread_ids=%s)",
                total, len(run_ids), thread_ids,
            )
        finally:
            conn.close()
    except Exception as exc:
        _log.exception(
            "[CHECKPOINT_CLEANUP] delete_checkpoints_for_runs failed: %s", exc
        )

    return total


def cleanup_orphan_checkpoints(*, dry_run: bool = False) -> dict:
    """Delete checkpoint rows whose thread_id has no matching agent_runs row.

    This handles historical orphans from before hard_delete_conversation
    was taught to cascade-delete checkpoints.  Also serves as a safety-net
    for any future edge cases.

    Args:
        dry_run: If True, only report what would be deleted.

    Returns:
        Summary dict with keys:
            orphan_thread_ids, deleted_rows, dry_run, errors
    """
    summary: dict = {
        "dry_run": dry_run,
        "orphan_thread_ids": [],
        "deleted_rows": 0,
        "errors": [],
    }

    conn_string = _pg_conn_string()
    if not conn_string:
        summary["errors"].append("no database connection string")
        return summary

    try:
        import psycopg

        # ── 1. Find orphan thread_ids ──────────────────────────────
        conn = psycopg.connect(conn_string)
        try:
            cur = conn.cursor()
            # Collect all distinct thread_ids from checkpoints that look like "run:N"
            cur.execute(
                "SELECT DISTINCT thread_id FROM checkpoints "
                "WHERE thread_id LIKE 'run:%'"
            )
            all_tids = [row[0] for row in cur.fetchall()]
            if not all_tids:
                _log.info(
                    "[CHECKPOINT_CLEANUP] orphan scan: no run-prefixed "
                    "thread_ids found"
                )
                return summary

            # Extract run_ids
            orphan_tids: list[str] = []
            for tid in all_tids:
                try:
                    rid = int(tid.split(":", 1)[1])
                except (IndexError, ValueError):
                    continue
                cur.execute(
                    "SELECT 1 FROM agent_runs WHERE id = %s LIMIT 1",
                    (rid,),
                )
                if cur.fetchone() is None:
                    orphan_tids.append(tid)

            summary["orphan_thread_ids"] = orphan_tids

            if not orphan_tids:
                _log.info(
                    "[CHECKPOINT_CLEANUP] orphan scan: no orphans found "
                    "(%d total run-prefixed thread_ids)",
                    len(all_tids),
                )
                return summary

            _log.info(
                "[CHECKPOINT_CLEANUP] orphan scan: found %d orphan "
                "thread_ids out of %d total",
                len(orphan_tids), len(all_tids),
            )

            # ── 2. Delete orphan data ──────────────────────────────
            for tid in orphan_tids:
                if dry_run:
                    for table in CHECKPOINT_DATA_TABLES:
                        cur.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE thread_id = %s",
                            (tid,),
                        )
                        row = cur.fetchone()
                        if row:
                            summary["deleted_rows"] += row[0]
                else:
                    rows_before = summary["deleted_rows"]
                    for table in CHECKPOINT_DATA_TABLES:
                        cur.execute(
                            f"DELETE FROM {table} WHERE thread_id = %s",
                            (tid,),
                        )
                        summary["deleted_rows"] += cur.rowcount
                    _log.info(
                        "[CHECKPOINT_CLEANUP] orphan cleanup: deleted %d rows "
                        "for thread_id=%s",
                        summary["deleted_rows"] - rows_before, tid,
                    )

            if not dry_run:
                conn.commit()
                _log.info(
                    "[CHECKPOINT_CLEANUP] orphan cleanup committed: "
                    "%d total rows deleted across %d thread_ids",
                    summary["deleted_rows"], len(orphan_tids),
                )
        finally:
            conn.close()
    except Exception as exc:
        _log.exception("[CHECKPOINT_CLEANUP] orphan cleanup failed")
        summary["errors"].append(str(exc))

    return summary
