"""Opt-in real Postgres checkpoints; every test owns a disposable schema."""

import os
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from src.web_app.tests.test_chat_control import env as env
from src.web_app.tests.test_supervisor_loop import configure, state
from src.web_app.agent.runtime.graph_builder import build_graph
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.mcp.local_provider import local_provider


pytestmark = pytest.mark.skipif(
    os.environ.get("SUPERVISOR_POSTGRES_TEST") != "1",
    reason="Opt-in isolated Postgres integration",
)


@pytest.fixture
def postgres_namespace():
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from src.web_app.core.config import settings

    conninfo = (
        settings.agent_checkpointer_database_url or settings.database_url
    ).replace("+psycopg2", "")
    schema = "test_supervisor_" + uuid4().hex
    with psycopg.connect(conninfo, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            yield make_conninfo(conninfo, options=f"-c search_path={schema}")
        finally:
            assert schema.startswith("test_supervisor_") and len(schema) == 48
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approved", "rejected"])
async def test_durable_graph_rebuild_preserves_completed_actions(
    env, monkeypatch, postgres_namespace, decision
):
    from src.web_app.services.rag_service import rag_service
    from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter
    from src.web_app.research.schemas import ResearchResult

    counts = {"read": 0, "research": 0, "save": 0, "email": 0}

    def read(*args, **kwargs):
        counts["read"] += 1
        return {
            "answer": "资料 [E1]",
            "evidence": [{"evidence_id": "E1", "quote": "source", "document_id": 1}],
        }

    async def research(*args, **kwargs):
        counts["research"] += 1
        assert (
            kwargs["evidence"] and kwargs["runtime_config"]["configurable"]["thread_id"]
        )
        return ResearchResult(
            summary="research result",
            markdown_report="# Research",
            evidence=kwargs["evidence"],
        )

    def provider(db, user_id, name, inputs, run_id):
        if name == "artifact_mcp.create_text_artifact":
            counts["save"] += 1
            return {"artifact_id": 1, "title": "saved"}
        counts["email"] += 1
        return {"success": True, "sent": True}

    monkeypatch.setattr(rag_service, "ask", read)
    monkeypatch.setattr(OpenDeepResearchAdapter, "run_research", research)
    monkeypatch.setattr(local_provider, "call", provider)
    config = {"configurable": {"thread_id": f"run:{env.run}"}, "recursion_limit": 60}
    async with AsyncPostgresSaver.from_conn_string(postgres_namespace) as saver:
        await saver.setup()
        with env.factory() as db:
            nodes = SupervisorNodes(db, {})
            configure(
                nodes,
                [
                    {"action": "rag", "arguments": {"query": "topic"}},
                    {"action": "deep_research", "arguments": {"query": "topic"}},
                    {
                        "action": "artifact",
                        "arguments": {"title": "result", "content": "research result"},
                    },
                    {
                        "action": "tool",
                        "arguments": {
                            "name": "email.send",
                            "input": {
                                "to": "test@example.test",
                                "subject": "result",
                                "body": "research",
                            },
                        },
                    },
                ],
                monkeypatch,
            )
            paused = await build_graph(nodes, saver).ainvoke(
                {
                    **state(env),
                    "research_authorized": True,
                    "save_policy": {"save_artifact": True},
                },
                config,
            )
            payload = paused["__interrupt__"][0].value
            assert counts == {"read": 1, "research": 1, "save": 1, "email": 0}
            approval = ApprovalRepository(db).get_by_user(
                env.user, payload["approval_id"]
            )
            ApprovalRepository(db).decide_pending(approval, decision, approval.payload)
    # Close the original saver/connection, rebuild all runtime objects.
    async with AsyncPostgresSaver.from_conn_string(postgres_namespace) as saver:
        with env.factory() as db:
            nodes = SupervisorNodes(db, {})
            configure(nodes, [{"action": "respond"}], monkeypatch)
            result = await build_graph(nodes, saver).ainvoke(
                Command(
                    resume={
                        "action": decision,
                        "approval_id": payload["approval_id"],
                        "tool_call_id": payload["tool_call_id"],
                    }
                ),
                config,
            )
            assert result["status"] == "completed"
            assert counts == {
                "read": 1,
                "research": 1,
                "save": 1,
                "email": int(decision == "approved"),
            }
            assert result["runtime_budget"] == {
                "steps": 5,
                "tool_calls": 2,
                "deep_research_calls": 1,
                "consecutive_failures": 0,
            }
