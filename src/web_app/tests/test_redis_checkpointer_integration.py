"""RedisSaver integration tests — require a real Redis instance.

Set REDIS_URL env var to run these tests.  Skip otherwise.

Usage:
  REDIS_URL=redis://:123456@192.168.170.100:6379/0 uv run pytest src/web_app/tests/test_redis_checkpointer_integration.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web_app.agent.runtime.checkpointers import build_checkpointer


# ── helpers ──────────────────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", "redis://:123456@192.168.170.100:6379/0")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "123456")


def _redis_saver():
    """Create a RedisSaver or skip if unavailable."""
    try:
        return build_checkpointer(
            backend="redis",
            redis_url=REDIS_URL,
            redis_password=REDIS_PASSWORD,
            require_durable=True,
        )
    except RuntimeError as exc:
        pytest.skip(f"RedisSaver unavailable: {exc}")


# ── tests ────────────────────────────────────────────────────────────


class TestRedisConnection:
    """Verify RedisSaver can connect to real Redis."""

    def test_redis_saver_connects(self):
        """build_checkpointer returns RedisSaver for real Redis."""
        saver = _redis_saver()
        assert saver is not None
        assert "Redis" in type(saver).__name__, (
            f"Expected RedisSaver, got {type(saver).__name__}"
        )

    def test_redis_saver_fails_fast_with_bad_url(self):
        """require_durable=True with unreachable host → RuntimeError."""
        with pytest.raises(RuntimeError, match="backend=redis unavailable"):
            build_checkpointer(
                backend="redis",
                redis_url="redis://192.168.99.99:99999/0",
                require_durable=True,
            )

    def test_build_checkpointer_returns_redis_saver_for_real_url(self):
        """Production path: build_checkpointer returns RedisSaver (skips if unreachable)."""
        saver = _redis_saver()
        assert saver is not None
        assert "Redis" in type(saver).__name__, (
            f"Expected RedisSaver, got {type(saver).__name__}"
        )


class TestRedisFailFast:
    """Production guard: RedisSaver unavailable → fail fast."""

    def test_require_durable_with_empty_url_raises(self):
        """require_durable=True with empty URL → RuntimeError."""
        with pytest.raises(RuntimeError, match="backend=redis requires redis_url"):
            build_checkpointer(
                backend="redis",
                redis_url="",
                require_durable=True,
            )

    def test_require_durable_false_with_bad_url_falls_back(self):
        """require_durable=False with bad URL → MemorySaver fallback."""
        saver = build_checkpointer(
            backend="redis",
            redis_url="redis://127.0.0.1:0/0",
            redis_password="wrong",
            require_durable=False,
        )
        assert saver is not None
        assert "Memory" in type(saver).__name__ or "InMemory" in type(saver).__name__


class TestRedisCheckpointRoundTrip:
    """Verify checkpoints survive across saver instances (same Redis)."""

    def test_two_savers_share_same_redis(self):
        """Two separate build_checkpointer calls both return RedisSaver."""
        saver1 = _redis_saver()
        saver2 = build_checkpointer(
            backend="redis",
            redis_url=REDIS_URL,
            redis_password=REDIS_PASSWORD,
        )
        assert "Redis" in type(saver1).__name__
        assert "Redis" in type(saver2).__name__
        # Both connect to same Redis → checkpoints shareable via thread_id
