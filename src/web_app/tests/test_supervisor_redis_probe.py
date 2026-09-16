"""Opt-in isolated Redis probe documents why this adapter is not V2-ready."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SUPERVISOR_REDIS_TEST") != "1", reason="Isolated Redis integration")


def test_installed_redis_adapter_cannot_provide_v2_async_recovery(tmp_path):
    from redis import Redis
    from langgraph.checkpoint.redis import RedisSaver
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from src.web_app.agent.runtime.checkpointers import build_checkpointer

    binary = Path(os.environ.get("REDIS_TEST_BIN", "D:/software/dabaredis/redis/redis-server.exe"))
    assert binary.is_file()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (tmp_path / "redis.log").open("w") as log:
        process = subprocess.Popen([str(binary), "--bind", "127.0.0.1", "--port", str(port),
            "--save", "", "--appendonly", "no", "--dir", str(tmp_path)], stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        client = Redis(host="127.0.0.1", port=port, socket_timeout=1)
        try:
            for _ in range(50):
                try:
                    if client.ping():
                        break
                except Exception:
                    time.sleep(0.1)
            assert client.ping(), "Owned Redis failed to start"
            assert RedisSaver.aget_tuple is BaseCheckpointSaver.aget_tuple
            # The locally installed server cannot supply RedisJSON/Search.
            with pytest.raises(RuntimeError, match="backend=redis unavailable"):
                build_checkpointer(backend="redis", redis_url=f"redis://127.0.0.1:{port}", require_durable=True)
        finally:
            client.close()
            process.terminate()
            process.wait(timeout=10)
