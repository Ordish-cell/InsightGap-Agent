"""Best-effort single-process wakeups. The database remains authoritative."""
import asyncio
from contextlib import contextmanager
from threading import RLock

_listeners = {}
_lock = RLock()


@contextmanager
def subscribe(run_id):
    entry = (asyncio.get_running_loop(), asyncio.Event())
    with _lock:
        _listeners.setdefault(run_id, set()).add(entry)
    try:
        yield entry[1]
    finally:
        with _lock:
            _listeners[run_id].discard(entry)
            if not _listeners[run_id]:
                del _listeners[run_id]


def notify(run_id):
    with _lock:
        listeners = tuple(_listeners.get(run_id, ()))
    for loop, signal in listeners:
        try:
            loop.call_soon_threadsafe(signal.set)
        except RuntimeError:
            pass  # A closed subscriber cannot affect a committed event.
