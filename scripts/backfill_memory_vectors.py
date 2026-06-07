"""Backfill PostgreSQL memories into Qdrant memory_vectors — with consistency checks.

Usage:
    uv run python scripts/backfill_memory_vectors.py --dry-run
    uv run python scripts/backfill_memory_vectors.py --check-only
    uv run python scripts/backfill_memory_vectors.py --user-id 2
    uv run python scripts/backfill_memory_vectors.py --memory-types semantic,episodic
    uv run python scripts/backfill_memory_vectors.py --force
    uv run python scripts/backfill_memory_vectors.py --limit 100
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.web_app.core.config import settings
from src.web_app.db.session import SessionLocal
from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.memory.qdrant_memory_store import QdrantMemoryStore


def parse_memory_types(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [t.strip().lower() for t in raw.split(",") if t.strip()]
    return items if items else None


def _preview(content: str, n: int = 60) -> str:
    text = content.replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def run_check_only(store: QdrantMemoryStore, repo: MemoryRepository, args) -> None:
    """Print a consistency report between PostgreSQL and Qdrant."""
    memory_types = parse_memory_types(args.memory_types) or ["semantic", "episodic"]
    include_working = args.include_working

    pg_total = repo.count_for_vector_backfill(
        user_id=args.user_id,
        memory_types=memory_types,
        include_working=include_working,
    )

    # Per-type PG counts
    pg_by_type: dict[str, int] = {}
    for mt in memory_types:
        pg_by_type[mt] = repo.count_for_vector_backfill(
            user_id=args.user_id,
            memory_types=[mt],
            include_working=False,
        )

    # Working count
    working_count = 0
    if not include_working:
        from sqlalchemy import func, select
        from src.web_app.models.orm import Memory
        db_work = repo.db
        stmt = (
            select(func.count(Memory.id))
            .where(
                Memory.memory_type == "working",
                Memory.content.isnot(None),
                Memory.content != "",
            )
        )
        if args.user_id is not None:
            stmt = stmt.where(Memory.user_id == args.user_id)
        working_count = int(db_work.execute(stmt).scalar() or 0)

    qdrant_total = store.count_indexed_memories(
        user_id=args.user_id,
        memory_types=memory_types,
    )

    qdrant_ids = store.list_indexed_memory_ids(
        user_id=args.user_id,
        memory_types=memory_types,
    )

    candidates = repo.list_for_vector_backfill(
        user_id=args.user_id,
        memory_types=memory_types,
        include_working=include_working,
    )

    pg_ids = {str(m.id) for m in candidates}
    missing_ids = pg_ids - qdrant_ids
    extra_ids = qdrant_ids - pg_ids

    print()
    print("=" * 60)
    print("  Memory Vector Index Consistency Report")
    print("=" * 60)
    print(f"  PostgreSQL eligible memories : {pg_total}")
    for mt, cnt in sorted(pg_by_type.items()):
        print(f"    {mt:20s}: {cnt}")
    print(f"    working excluded           : {working_count}")
    print()
    print(f"  Qdrant indexed memories      : {qdrant_total}")
    print()
    print(f"  Missing in Qdrant            : {len(missing_ids)}")
    if missing_ids:
        for mid in sorted(missing_ids, key=int)[:20]:
            match = [m for m in candidates if str(m.id) == mid]
            if match:
                m = match[0]
                print(f"    memory_id={mid} type={m.memory_type:10s} "
                      f"user_id={m.user_id} content={_preview(m.content, 55)}")
    if len(missing_ids) > 20:
        print(f"    ... and {len(missing_ids) - 20} more")
    print()
    print(f"  Extra in Qdrant (orphan)     : {len(extra_ids)}")
    if extra_ids:
        for eid in sorted(extra_ids)[:10]:
            print(f"    memory_id={eid} (not in PostgreSQL)")
    if len(extra_ids) > 10:
        print(f"    ... and {len(extra_ids) - 10} more")
    print("=" * 60)
    print("[CHECK-ONLY DONE]")


def run_backfill(store: QdrantMemoryStore, repo: MemoryRepository, args) -> None:
    """Upsert eligible memories into Qdrant."""
    memory_types = parse_memory_types(args.memory_types) or ["semantic", "episodic"]
    include_working = args.include_working

    candidates = repo.list_for_vector_backfill(
        user_id=args.user_id,
        memory_types=memory_types,
        include_working=include_working,
        limit=args.limit if args.limit else None,
    )

    stats = {
        "total_candidates": len(candidates),
        "skipped_existing": 0,
        "skipped_empty": 0,
        "skipped_working": 0,
        "upserted": 0,
        "failed": 0,
    }
    errors: list[str] = []

    # Bulk-load existing Qdrant memory_ids for dedup and force-cleanup
    qdrant_indexed_ids = store.list_indexed_memory_ids(
        user_id=args.user_id,
        memory_types=memory_types,
    )
    print(f"[INFO] Pre-loaded {len(qdrant_indexed_ids)} indexed memory IDs from Qdrant")

    for i, mem in enumerate(candidates, start=1):
        content = (mem.content or "").strip()
        if not content:
            stats["skipped_empty"] += 1
            if args.dry_run:
                print(f"[SKIP-EMPTY] #{i}/{len(candidates)} id={mem.id}")
            continue

        if mem.memory_type == "working" and not include_working:
            stats["skipped_working"] += 1
            if args.dry_run:
                print(f"[SKIP-WORK]  #{i}/{len(candidates)} id={mem.id} type=working")
            continue

        # ── Dedup check ──────────────────────────────────────────
        if not args.force and str(mem.id) in qdrant_indexed_ids:
            stats["skipped_existing"] += 1
            if args.dry_run:
                print(f"[SKIP-EXIST] #{i}/{len(candidates)} id={mem.id} "
                      f"type={mem.memory_type}")
            continue

        if args.dry_run:
            print(f"[DRY-RUN]    #{i}/{len(candidates)} id={mem.id} "
                  f"type={mem.memory_type:10s} importance={mem.importance:.2f} "
                  f"content={_preview(content)}")
            stats["upserted"] += 1
            continue

        # ── Force: delete stale points before re-upsert ───────
        if args.force and str(mem.id) in qdrant_indexed_ids:
            try:
                store.delete_by_memory_id(mem.id)
            except Exception:
                pass

        # ── Upsert ────────────────────────────────────────────────
        try:
            point_id = store.upsert_memory(
                memory_id=mem.id,
                user_id=mem.user_id,
                content=content,
                memory_type=mem.memory_type,
                importance=float(mem.importance or 0),
                source_type=mem.source_type or "",
                metadata=mem.metadata_json or {},
            )
            # Update PG metadata (best-effort)
            try:
                db2 = SessionLocal()
                try:
                    from src.web_app.models.orm import Memory
                    MemoryRepository(db2).update(
                        db2.get(Memory, mem.id),
                        qdrant_point_id=point_id,
                        metadata_json={
                            **(mem.metadata_json or {}),
                            "qdrant_indexed": True,
                            "qdrant_indexed_at": datetime.now(UTC).isoformat(),
                        },
                    )
                finally:
                    db2.close()
            except Exception:
                pass
            stats["upserted"] += 1
            if i % 20 == 0:
                print(f"[PROGRESS] {i}/{len(candidates)} — {stats['upserted']} ok, "
                      f"{stats['failed']} fail, {stats['skipped_existing']} exist, "
                      f"{stats['skipped_working']} work, {stats['skipped_empty']} empty")
        except Exception as exc:
            stats["failed"] += 1
            err_msg = f"id={mem.id}: {exc}"
            errors.append(err_msg)
            print(f"[FAIL] #{i}/{len(candidates)} {err_msg}")

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Memory Vector Backfill Report")
    print("=" * 60)
    if args.dry_run:
        print("  MODE: DRY-RUN (no writes)")
    if args.user_id:
        print(f"  User filter        : user_id={args.user_id}")
    print(f"  Memory types       : {memory_types}")
    print(f"  Include working    : {include_working}")
    print(f"  Force reindex      : {args.force}")
    print(f"  Candidates         : {stats['total_candidates']}")
    print(f"  Upserted           : {stats['upserted']}")
    print(f"  Skipped (existing) : {stats['skipped_existing']}")
    print(f"  Skipped (empty)    : {stats['skipped_empty']}")
    print(f"  Skipped (working)  : {stats['skipped_working']}")
    print(f"  Failed             : {stats['failed']}")
    if not args.dry_run:
        points_after = store.count_indexed_memories(
            user_id=args.user_id,
            memory_types=memory_types,
        )
        print(f"  Qdrant points now  : {points_after}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
    print("=" * 60)
    print("[DONE]" if not args.dry_run else "[DRY-RUN DONE]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill PostgreSQL memories into Qdrant memory_vectors"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing to Qdrant")
    parser.add_argument("--check-only", action="store_true",
                        help="Only print consistency report, no upserts")
    parser.add_argument("--user-id", type=int, default=None,
                        help="Only process this user_id")
    parser.add_argument("--memory-types", type=str, default="semantic,episodic",
                        help="Comma-separated memory types (default: semantic,episodic)")
    parser.add_argument("--include-working", action="store_true",
                        help="Also backfill working memories (default: skip)")
    parser.add_argument("--force", action="store_true",
                        help="Re-upsert even if already indexed in Qdrant")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max memories to process (for debugging)")
    args = parser.parse_args()

    if not settings.qdrant_url:
        print("[ERROR] QDRANT_URL is not configured — aborting.")
        sys.exit(1)

    print(f"[INFO] Qdrant URL            : {settings.qdrant_url}")
    print(f"[INFO] Qdrant collection     : {settings.memory_qdrant_collection}")
    print(f"[INFO] Vector size           : {settings.qdrant_vector_size}")
    print(f"[INFO] Dry run               : {args.dry_run}")
    print(f"[INFO] Check only            : {args.check_only}")
    print(f"[INFO] Force                 : {args.force}")
    print()

    store = QdrantMemoryStore()
    store.ensure_collection()

    db = SessionLocal()
    try:
        repo = MemoryRepository(db)

        if args.check_only:
            run_check_only(store, repo, args)
            return

        run_backfill(store, repo, args)
    finally:
        db.close()


if __name__ == "__main__":
    main()
