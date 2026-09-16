"""Serialize a run's resume across workers without a permanent crash lease."""

from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
from pathlib import Path
from tempfile import gettempdir

from sqlalchemy import text


@contextmanager
def claim_resume(db, user_id, run_id):
    bind = db.get_bind()
    if bind.dialect.name == "postgresql":
        # Session locks disappear when the connection/process dies. Hold a
        # dedicated connection because application repositories commit often.
        with bind.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            key = int.from_bytes(
                sha256(f"insightgap:resume:{user_id}:{run_id}".encode()).digest()[:8],
                "big",
                signed=True,
            )
            claimed = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
                ).scalar()
            )
            try:
                yield claimed
            finally:
                if claimed:
                    connection.execute(
                        text("SELECT pg_advisory_unlock(:key)"), {"key": key}
                    )
    else:
        from filelock import FileLock, Timeout

        database = str(getattr(bind, "url", id(bind)))
        key = sha256(f"{database}:{user_id}:{run_id}".encode()).hexdigest()
        directory = Path(gettempdir()) / "insightgap-resume-locks"
        directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(directory / f"{key}.lock")
        try:
            lock.acquire(timeout=0)
        except Timeout:
            yield False
        else:
            try:
                yield True
            finally:
                lock.release()


def exclusive_resume(function):
    @wraps(function)
    async def wrapped(db, user_id, run_id, *args, **kwargs):
        with claim_resume(db, user_id, run_id) as claimed:
            if not claimed:
                return {"run_id": run_id, "status": "resuming", "already_running": True}
            # A new session may have read the pre-lock snapshot.
            db.expire_all()
            return await function(db, user_id, run_id, *args, **kwargs)

    return wrapped
