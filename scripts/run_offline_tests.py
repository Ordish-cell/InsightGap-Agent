"""Run repository tests with disposable storage, SQLite, and no outbound sockets.

Provider and durable checkpoint integration tests run separately. Individual
tests remain responsible for supplying their fake model/provider responses.
"""

from pathlib import Path
import os
import shutil
import socket
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    import pytest

    directory = Path(tempfile.mkdtemp(prefix="insightgap-offline-tests-"))
    print(f"Isolated test workspace: {directory}", flush=True)
    args = sys.argv[1:] or ["src/web_app/tests", "tests", "-q", "--tb=short"]
    args = [str(ROOT / arg.split("::", 1)[0]) + ("::" + arg.split("::", 1)[1] if "::" in arg else "")
            if (ROOT / arg.split("::", 1)[0]).exists() else arg for arg in args]
    sys.path.insert(0, str(ROOT))
    # Several older tests and services use relative storage paths.
    for relative in ("alembic", "src/web_app/tests/fixtures"):
        shutil.copytree(
            ROOT / relative,
            directory / relative,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    os.chdir(directory)
    from cryptography.fernet import Fernet

    os.environ.update(
        MODEL_CREDENTIALS_ENCRYPTION_KEY=Fernet.generate_key().decode(),
        AGENT_CHECKPOINTER_BACKEND="memory",
        AGENT_CHECKPOINTER_REQUIRE_DURABLE="false",
        AGENT_CHECKPOINTER_DATABASE_URL="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CHAT_REAL_SMOKE="0",
        DOCUMENT_REAL_SMOKE="0",
    )
    from src.web_app.core.config import Settings, settings

    Settings.database_url = property(
        lambda self: f"sqlite:///{directory / 'database.db'}"
    )
    settings.artifact_storage_path = str(directory / "artifacts")
    settings.local_tools_workspace_dir = str(directory / "workspace")
    settings.agent_checkpointer_backend = "memory"
    settings.agent_checkpointer_require_durable = False
    settings.qdrant_url = ""
    settings.enable_neo4j = False
    settings.redis_url = ""
    from src.web_app.db.session import engine
    from src.web_app.db.base import Base
    from src.web_app.models import orm  # noqa: F401

    Base.metadata.create_all(engine)

    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def no_network(*args, **kwargs):
        caller = sys._getframe(1).f_code
        if (
            caller.co_filename == socket.__file__
            and caller.co_name == "_fallback_socketpair"
        ):
            # Windows implements asyncio's private wakeup pipe over loopback.
            return original_connect(*args, **kwargs)
        raise OSError(
            "Outbound network disabled by run_offline_tests; supply a fake provider"
        )

    socket.getaddrinfo = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network

    # libpq bypasses Python sockets. Offline tests must not reach development PG.
    import psycopg
    import psycopg2
    native_guard = pytest.MonkeyPatch()
    def no_postgres(*args, **kwargs):
        raise OSError("PostgreSQL disabled in offline suite; use isolated integration tests")
    async def no_async_postgres(*args, **kwargs):
        return no_postgres()
    native_guard.setattr(psycopg, "connect", no_postgres)
    native_guard.setattr(psycopg.Connection, "connect", no_postgres)
    native_guard.setattr(psycopg.AsyncConnection, "connect", no_async_postgres)
    native_guard.setattr(psycopg2, "connect", no_postgres)
    # DDGS uses a native HTTP client, which also bypasses Python sockets.
    from duckduckgo_search import DDGS
    native_guard.setattr(DDGS, "text", no_network)

    class SeparateIntegration:
        def pytest_runtest_logreport(self, report):
            import json
            with (directory / "progress.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps({"nodeid": report.nodeid, "when": report.when,
                    "outcome": report.outcome, "duration": report.duration,
                    "failure": str(report.longrepr) if report.failed else None}) + "\n")

        def pytest_collection_modifyitems(self, items):
            for item in items:
                external = item.path.name in {
                    "test_supervisor_durable.py",
                    "test_supervisor_process_restart.py",
                    "test_supervisor_redis_probe.py",
                }
                external |= item.path.name == "test_postgres_checkpoint_e2e.py" and item.name != "test_require_durable_blocks_memory"
                external |= item.path.name == "test_redis_checkpointer_integration.py" and item.name in {
                    "test_redis_saver_connects", "test_build_checkpointer_returns_redis_saver_for_real_url",
                    "test_two_savers_share_same_redis",
                }
                external |= item.path.name == "test_e2e_redis_restart_approval.py" and item.name == "test_redis_saver_can_connect"
                if external:
                    item.add_marker(
                        pytest.mark.skip(
                            reason="Durable backend integration is run separately with isolated namespaces"
                        )
                    )

    try:
        return pytest.main(
            [*args, f"--junitxml={directory / 'results.xml'}"],
            plugins=[SeparateIntegration()],
        )
    finally:
        native_guard.undo()
        socket.getaddrinfo = original_getaddrinfo
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
