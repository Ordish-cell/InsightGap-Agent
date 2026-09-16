"""Real process and owned PostgreSQL service restart; no development DB access."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from collections import Counter

import pytest

from src.web_app.tests.test_chat_control import env as env
from src.web_app.tests.test_supervisor_loop import state

pytestmark = pytest.mark.skipif(os.environ.get("SUPERVISOR_PROCESS_TEST") != "1", reason="Isolated process integration")


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_process_and_postgres_restart_preserves_actions(env, tmp_path, decision):
    binaries = Path(os.environ.get("POSTGRES_TEST_BIN", "D:/software/pgsql/bin"))
    assert (binaries / "pg_ctl.exe").is_file(), "Set POSTGRES_TEST_BIN to an installed PostgreSQL bin directory"
    data = tmp_path / "postgres"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    def command(*args):
        # PostgreSQL's Windows grandchildren can inherit pipe handles. A file
        # prevents communicate() from waiting for the entire server lifetime.
        output = tmp_path / "command.log"
        with output.open("w") as handle:
            result = subprocess.run([str(a) for a in args], stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, timeout=60, creationflags=flags)
        assert result.returncode == 0, output.read_text(errors="replace")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command(binaries / "initdb.exe", "-D", data, "-A", "trust", "-U", "test", "--no-locale", "-E", "UTF8")
    def start():
        command(binaries / "pg_ctl.exe", "-D", data, "-l", tmp_path / "postgres.log", "-o", f"-h 127.0.0.1 -p {port}", "-w", "start")
    start()
    try:
        with env.factory() as db:
            database = str(db.get_bind().url)
        payload = {"database": database, "postgres": f"host=127.0.0.1 port={port} user=test dbname=postgres",
            "state": {**state(env), "research_authorized": True, "save_policy": {"save_artifact": True}},
            "config": {"configurable": {"thread_id": f"run:{env.run}"}, "recursion_limit": 60}}
        (tmp_path / "input.json").write_text(json.dumps(payload))
        command(sys.executable, "-m", "src.web_app.tests.supervisor_process_worker", tmp_path, "start", decision)
        # Crash only the instance created above, after the approval checkpoint is durable.
        command(binaries / "pg_ctl.exe", "-D", data, "-m", "immediate", "-w", "stop")
        start()
        command(sys.executable, "-m", "src.web_app.tests.supervisor_process_worker", tmp_path, "resume", decision)
        result = json.loads((tmp_path / "result.json").read_text())
        counts = Counter(json.loads(line)["kind"] for line in (tmp_path / "receipts.jsonl").read_text().splitlines())
        assert counts == Counter(read=1, research=1, save=1, email=int(decision == "approved"))
        assert result == {"status": "completed", "budget": {"steps": 5, "tool_calls": 2, "deep_research_calls": 1, "consecutive_failures": 0}}
    finally:
        command(binaries / "pg_ctl.exe", "-D", data, "-m", "fast", "-w", "stop")


def test_process_crash_after_external_effect_does_not_retry(env, tmp_path):
    """Kill the worker after a provider receipt, before the local result commit."""
    from src.web_app.tests.test_supervisor_recovery import prepare, ARGS

    call_id = prepare(env)
    with env.factory() as db:
        database = str(db.get_bind().url)
    payload = {"database": database, "user": env.user, "run": env.run,
               "call": call_id, "args": ARGS, "receipt": str(tmp_path / "external-receipt"),
               "result": str(tmp_path / "unknown-result.json")}
    input_file = tmp_path / "crash-input.json"
    input_file.write_text(json.dumps(payload))
    worker = tmp_path / "crash_worker.py"
    worker.write_text("""
import json, os, sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.web_app.mcp.tool_executor import tool_executor
from src.web_app.mcp.local_provider import local_provider
p = json.loads(Path(sys.argv[1]).read_text())
def provider(*args, **kwargs):
    if sys.argv[2] == 'resume':
        os._exit(99)
    with open(p['receipt'], 'w') as receipt:
        receipt.write('external side effect completed')
        receipt.flush()
        os.fsync(receipt.fileno())
    os._exit(71)
local_provider.call = provider
with Session(create_engine(p['database'])) as db:
    result = tool_executor.execute_approved_tool_once(
        db, p['user'], p['call'], 'email.send', p['args'], p['run'])
    Path(p['result']).write_text(json.dumps(result))
""")
    worker_env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    def run(mode):
        return subprocess.run([sys.executable, str(worker), str(input_file), mode],
            capture_output=True, text=True, timeout=60, env=worker_env)
    crashed = run("crash")
    assert crashed.returncode == 71, crashed.stderr
    assert Path(payload["receipt"]).read_text() == "external side effect completed"
    assert not Path(payload["result"]).exists()
    resumed = run("resume")
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(Path(payload["result"]).read_text())
    assert result["error_code"] == "TOOL_OUTCOME_UNKNOWN"
    assert result["success"] is False
