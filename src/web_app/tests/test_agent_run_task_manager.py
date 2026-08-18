import asyncio

import pytest

from src.web_app.services.agent_run_task_manager import AgentRunTaskManager


@pytest.mark.asyncio
async def test_task_manager_deduplicates_active_run(monkeypatch):
    manager = AgentRunTaskManager()
    release = asyncio.Event()

    async def fake_run(run_id, user_id, payload, *, resume):
        await release.wait()

    monkeypatch.setattr(manager, "_run", fake_run)

    assert manager.start(42, 7, payload={"user_input": "test"}) is True
    assert manager.start(42, 7, payload={"user_input": "test"}) is False
    assert manager.is_running(42) is True

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert manager.is_running(42) is False


@pytest.mark.asyncio
async def test_task_manager_shutdown_cancels_owned_tasks(monkeypatch):
    manager = AgentRunTaskManager()
    cancelled = asyncio.Event()

    async def fake_run(run_id, user_id, payload, *, resume):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(manager, "_run", fake_run)
    manager.start(9, 3, resume=True)
    await asyncio.sleep(0)

    await manager.shutdown()

    assert cancelled.is_set()
    assert manager.is_running(9) is False
