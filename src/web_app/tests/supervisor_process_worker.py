"""Isolated subprocess participant; all external capabilities are simulated."""
import asyncio
import json
import sys
import os
from pathlib import Path


async def main():
    import pytest
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.types import Command
    from src.web_app.agent.runtime.graph_builder import build_graph
    from src.web_app.agent.runtime.nodes import SupervisorNodes
    from src.web_app.tests.test_supervisor_loop import configure
    from src.web_app.db.repositories.approval_repository import ApprovalRepository
    from src.web_app.mcp.local_provider import local_provider
    from src.web_app.services.rag_service import rag_service
    from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter
    from src.web_app.research.schemas import ResearchResult

    directory, mode, decision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    data = json.loads((directory / "input.json").read_text())
    engine = create_engine(data["database"])
    factory = sessionmaker(bind=engine)
    def receipt(kind):
        with (directory / "receipts.jsonl").open("a") as handle:
            handle.write(json.dumps({"kind": kind}) + "\n")
    def read(*a, **kw):
        receipt("read")
        return {"answer": "source", "evidence": [{"quote": "source", "evidence_id": "E1"}]}
    async def research(*a, **kw):
        receipt("research")
        return ResearchResult(summary="research", markdown_report="research", evidence=[])
    def provider(db, user_id, name, inputs, run_id):
        receipt("save" if name == "artifact_mcp.create_text_artifact" else "email")
        return {"artifact_id": 1} if name == "artifact_mcp.create_text_artifact" else {"success": True, "sent": True}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rag_service, "ask", read)
        patch.setattr(OpenDeepResearchAdapter, "run_research", research)
        patch.setattr(local_provider, "call", provider)
        async with AsyncPostgresSaver.from_conn_string(data["postgres"]) as saver:
            await saver.setup()
            with factory() as db:
                nodes = SupervisorNodes(db, {})
                actions = [
                    {"action": "rag", "arguments": {"query": "topic"}},
                    {"action": "deep_research", "arguments": {"query": "topic"}},
                    {"action": "artifact", "arguments": {"title": "result", "content": "report"}},
                    {"action": "tool", "arguments": {"name": "email.send", "input": {
                        "to": "test@example.test", "subject": "test", "body": "test"}}},
                ] if mode == "start" else [{"action": "respond"}]
                configure(nodes, actions, patch)
                graph = build_graph(nodes, saver)
                if mode == "start":
                    result = await graph.ainvoke(data["state"], data["config"])
                    output = result["__interrupt__"][0].value
                else:
                    payload = json.loads((directory / "pause.json").read_text())
                    approval = ApprovalRepository(db).get_by_user(data["state"]["user_id"], payload["approval_id"])
                    ApprovalRepository(db).decide_pending(approval, decision, approval.payload)
                    result = await graph.ainvoke(Command(resume={"action": decision,
                        "approval_id": payload["approval_id"], "tool_call_id": payload["tool_call_id"]}), data["config"])
                    output = {"status": result["status"], "budget": result["runtime_budget"]}
                (directory / ("pause.json" if mode == "start" else "result.json")).write_text(json.dumps(output))
    engine.dispose()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
