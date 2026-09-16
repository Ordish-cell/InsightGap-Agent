"""Provider side effects and run recovery contracts, using isolated storage."""

import asyncio
from threading import Event

import pytest

from src.web_app.tests.test_chat_control import env as env
from src.web_app.mcp.tool_executor import tool_executor
from src.web_app.mcp.local_provider import local_provider
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.mcp_repository import ToolCallRepository


ARGS = {"to": "test@example.test", "subject": "test", "body": "test"}


def prepare(env, *, dry_run=False):
    with env.factory() as db:
        call = tool_executor.call_tool(
            db,
            env.user,
            "email.send",
            ARGS,
            env.run,
            idempotency_key="test-action",
            dry_run=dry_run,
        )
        approval = ApprovalRepository(db).get_by_user(env.user, call.approval_id)
        ApprovalRepository(db).decide_pending(approval, "approved", approval.payload)
        return call.id


@pytest.mark.asyncio
async def test_concurrent_approved_execution_is_claimed_once(env, monkeypatch):
    call_id = prepare(env)
    entered, release = Event(), Event()
    calls = []

    def provider(*a, **kw):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return {"success": True, "sent": True}

    monkeypatch.setattr(local_provider, "call", provider)

    def execute():
        with env.factory() as db:
            return tool_executor.execute_approved_tool_once(
                db, env.user, call_id, "email.send", ARGS, env.run
            )

    first = asyncio.create_task(asyncio.to_thread(execute))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        racing = await asyncio.to_thread(execute)
        assert racing["error_code"] == "TOOL_OUTCOME_UNKNOWN"
    finally:
        release.set()
    result = await first
    replay = await asyncio.to_thread(execute)
    assert result == replay and result["sent"] is True
    assert calls == [1]


def test_approved_dry_run_never_enters_provider(env, monkeypatch):
    call_id = prepare(env, dry_run=True)
    monkeypatch.setattr(
        local_provider, "call", lambda *a, **kw: pytest.fail("dry run sent email")
    )
    with env.factory() as db:
        result = tool_executor.execute_approved_tool_once(
            db, env.user, call_id, "email.send", ARGS, env.run
        )
        assert result["dry_run"] is True


def test_resume_arguments_and_approval_snapshot_are_verified(env, monkeypatch):
    call_id = prepare(env)
    monkeypatch.setattr(
        local_provider,
        "call",
        lambda *a, **kw: pytest.fail("mismatched action executed"),
    )
    with env.factory() as db:
        with pytest.raises(ValueError, match="arguments mismatch"):
            tool_executor.execute_approved_tool_once(
                db,
                env.user,
                call_id,
                "email.send",
                {**ARGS, "body": "changed"},
                env.run,
            )
        approval = ApprovalRepository(db).get_by_idempotency_key("test-action")
        ApprovalRepository(db).update(
            approval,
            payload={
                **approval.payload,
                "tool_args": {**ARGS, "to": "changed@example.test"},
            },
        )
        with pytest.raises(ValueError, match="approval arguments mismatch"):
            tool_executor.execute_approved_tool_once(
                db, env.user, call_id, "email.send", ARGS, env.run
            )


def test_crashed_provider_attempt_is_not_retried(env, monkeypatch):
    call_id = prepare(env)
    monkeypatch.setattr(
        local_provider, "call", lambda *a, **kw: pytest.fail("unknown result retried")
    )
    with env.factory() as db:
        ToolCallRepository(db).claim_execution(call_id, env.user, ("waiting_approval",))
    with env.factory() as db:
        result = tool_executor.execute_approved_tool_once(
            db, env.user, call_id, "email.send", ARGS, env.run
        )
        assert result["error_code"] == "TOOL_OUTCOME_UNKNOWN"


def test_resume_lock_releases_and_coordinates_instances(env):
    from src.web_app.agent.runtime.resume_lock import claim_resume

    with env.factory() as first, env.factory() as second:
        with claim_resume(first, env.user, env.run) as claimed:
            assert claimed
            with claim_resume(second, env.user, env.run) as duplicate:
                assert not duplicate
        with claim_resume(second, env.user, env.run) as claimed:
            assert claimed
