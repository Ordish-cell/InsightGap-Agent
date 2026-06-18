"""Approval expiry — auto-expire stale pending approvals.

When a pending approval exceeds agent_approval_pending_ttl_hours (default 24h),
it is automatically expired.  Expired approvals:
- Cannot be approved or rejected (clear error returned)
- Have their run/message/tool_call status synced to expired/cancelled
- Have their checkpoints eligible for cleanup after the expired TTL (7 days)

Usage (cron / admin):
    from src.web_app.services.approval_expiry import expire_stale_approvals
    summary = expire_stale_approvals()
    print(summary)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

_log = logging.getLogger(__name__)


def expire_stale_approvals(
    *,
    ttl_hours: int | None = None,
    dry_run: bool = False,
    db=None,
) -> dict:
    """Find and expire all pending approvals older than the TTL.

    For each expired approval, syncs:
        approval.status -> "expired"
        run.status -> "expired" (if still waiting_approval/paused)
        message.status -> "expired" (if still waiting_approval)
        tool_call.status -> "expired" (if still waiting_approval)

    Args:
        ttl_hours: Override default TTL. None = use config value.
        dry_run: If True, only report what would be expired.
        db: Optional SQLAlchemy session (for test injection).

    Returns:
        Summary dict with keys:
            expired_count, expired_approval_ids, affected_run_ids,
            dry_run, errors
    """
    from src.web_app.core.config import settings

    if ttl_hours is None:
        ttl_hours = getattr(settings, "agent_approval_pending_ttl_hours", 24)

    summary: dict = {
        "dry_run": dry_run,
        "ttl_hours": ttl_hours,
        "expired_count": 0,
        "expired_approval_ids": [],
        "affected_run_ids": [],
        "errors": [],
    }

    try:
        from src.web_app.models.orm import Approval as _Approval
        from sqlalchemy import select as _select

        own_db = db is None
        if own_db:
            from src.web_app.db.session import SessionLocal
            db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=ttl_hours)
            cutoff_naive = cutoff.replace(tzinfo=None)

            # Find all pending approvals older than cutoff
            stmt = _select(_Approval).where(
                _Approval.status == "pending",
                _Approval.created_at < cutoff_naive,
            )
            stale_approvals = list(db.execute(stmt).scalars())

            if not stale_approvals:
                _log.info(
                    "[APPROVAL_EXPIRY] no stale approvals found "
                    "(ttl=%dh, cutoff=%s)", ttl_hours, cutoff_naive
                )
                return summary

            _log.info(
                "[APPROVAL_EXPIRY] found %d stale approvals (ttl=%dh)",
                len(stale_approvals), ttl_hours,
            )

            for approval in stale_approvals:
                approval_id = approval.id
                run_id = approval.run_id

                _log.info(
                    "[APPROVAL_EXPIRY] %s approval_id=%s run_id=%s "
                    "created_at=%s",
                    "DRY_RUN would expire" if dry_run else "expiring",
                    approval_id, run_id, approval.created_at,
                )

                if dry_run:
                    summary["expired_count"] += 1
                    summary["expired_approval_ids"].append(approval_id)
                    if run_id:
                        summary["affected_run_ids"].append(run_id)
                    continue

                # 1. Expire the approval
                approval.status = "expired"
                payload = dict(approval.payload or {})
                payload["expired_at"] = now.isoformat()
                payload["expired_reason"] = (
                    f"Approval exceeded TTL of {ttl_hours} hours"
                )
                approval.payload = payload
                db.add(approval)

                # 2. Sync run status
                if run_id:
                    from src.web_app.models.orm import AgentRun as _Run
                    run = db.execute(
                        _select(_Run).where(_Run.id == run_id)
                    ).scalar_one_or_none()
                    if run and run.status in ("waiting_approval", "paused"):
                        run.status = "expired"
                        gs = dict(run.graph_state or {})
                        gs["approval_required"] = False
                        gs["approval_payload"] = None
                        gs["error"] = ""
                        run.graph_state = gs
                        run.completed_at = datetime.now()
                        db.add(run)
                        _log.info(
                            "[APPROVAL_EXPIRY] run %s -> expired", run_id
                        )

                # 3. Sync message status
                try:
                    from src.web_app.models.orm import AgentChatMessage as _Msg
                    msg_stmt = _select(_Msg).where(
                        _Msg.run_id == run_id,
                        _Msg.role == "assistant",
                        _Msg.status == "waiting_approval",
                    )
                    msgs = list(db.execute(msg_stmt).scalars())
                    for msg in msgs:
                        msg.status = "expired"
                        db.add(msg)
                except Exception as exc:
                    _log.warning(
                        "[APPROVAL_EXPIRY] message sync failed for "
                        "run_id=%s: %s", run_id, exc
                    )

                # 4. Sync tool_call status
                try:
                    from src.web_app.models.orm import ToolCall as _TC
                    tc_stmt = _select(_TC).where(
                        _TC.run_id == run_id,
                        _TC.status == "waiting_approval",
                    )
                    tcs = list(db.execute(tc_stmt).scalars())
                    for tc in tcs:
                        tc.status = "expired"
                        db.add(tc)
                except Exception as exc:
                    _log.warning(
                        "[APPROVAL_EXPIRY] tool_call sync failed for "
                        "run_id=%s: %s", run_id, exc
                    )

                summary["expired_count"] += 1
                summary["expired_approval_ids"].append(approval_id)
                if run_id:
                    summary["affected_run_ids"].append(run_id)

            if not dry_run:
                db.commit()
                _log.info(
                    "[APPROVAL_EXPIRY] committed - %d approvals expired",
                    summary["expired_count"],
                )
        finally:
            if own_db:
                db.close()
    except Exception as exc:
        _log.exception("[APPROVAL_EXPIRY] failed")
        summary["errors"].append(str(exc))

    return summary
