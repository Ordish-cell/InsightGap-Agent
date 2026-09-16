"""Exercise the native graph across real processes and an owned PG restart."""
import os
import pytest
from src.web_app.tests.test_chat_control import env

pytestmark = pytest.mark.skipif(os.environ.get("NATIVE_PROCESS_TEST") != "1", reason="Opt-in native graph independent Postgres restart")


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_native_process_and_postgres_restart(env, tmp_path, decision, monkeypatch):
    from src.web_app.tests.test_supervisor_process_restart import test_process_and_postgres_restart_preserves_actions
    monkeypatch.setenv("NATIVE_PROCESS_TEST", "1")
    test_process_and_postgres_restart_preserves_actions(env, tmp_path, decision)


def test_native_shared_executor_unknown_effect_stops_retry(env, tmp_path):
    from src.web_app.tests.test_supervisor_process_restart import test_process_crash_after_external_effect_does_not_retry
    test_process_crash_after_external_effect_does_not_retry(env, tmp_path)
