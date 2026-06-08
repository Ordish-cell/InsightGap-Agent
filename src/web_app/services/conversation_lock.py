"""Per-conversation asyncio.Lock manager.

Ensures that runs within the same conversation are serialised (no race
conditions on shared conversation state), while runs belonging to
different conversations proceed concurrently.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class ConversationLockManager:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def acquire(self, conversation_id: str) -> asyncio.Lock:
        """Return the lock for *conversation_id*, creating it if needed.

        The caller should:
            lock = await manager.acquire(cid)
            async with lock:
                ...
        """
        async with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_id] = lock
            return lock

    async def release(self, conversation_id: str) -> None:
        """Remove the lock if it is no longer held — keeps the dict small."""
        async with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is not None and not lock.locked():
                self._locks.pop(conversation_id, None)

    def count(self) -> int:
        return len(self._locks)


# Singleton shared by all agent runs.
conversation_lock_manager = ConversationLockManager()
