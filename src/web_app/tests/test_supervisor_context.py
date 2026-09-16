"""Cross-turn consent and evidence contracts independent of model heuristics."""

import pytest

from src.web_app.tests.test_chat_control import env as env
from src.web_app.tests.test_supervisor_loop import state
from src.web_app.agent.runtime.nodes import SupervisorNodes
from src.web_app.agent.runtime.context import bootstrap
from src.web_app.agent.runtime.capabilities import observe, execute_capability
from src.web_app.agent.runtime.state import CapabilityResult


@pytest.mark.parametrize(
    "text,allowed",
    [
        ("什么是 Deep Research？", False),
        ("Deep Research 是什么？", False),
        ("解释一下深度研究和普通搜索的区别", False),
        ("如何使用深度研究？", False),
        ("请深入研究这个问题", True),
        ("启动深度研究：电池回收", True),
        ("Please conduct deep research on battery recycling", True),
        ("不要深入研究", False),
    ],
)
def test_research_mention_is_not_authorization(text, allowed):
    from src.web_app.agent.runtime.context import research_consent

    assert bool(research_consent(text, {}, None)) is allowed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,authorized",
    [("好的", True), ("不要", False), ("可能吧", False), ("解释哈希表", False)],
)
async def test_persisted_research_proposal_requires_specific_confirmation(
    env, monkeypatch, reply, authorized
):
    from src.web_app.models.orm import AgentChatMessage

    with env.factory() as db:
        db.add(
            AgentChatMessage(
                message_id="proposal",
                user_id=env.user,
                conversation_id="chat",
                role="assistant",
                content="要深入研究这个问题吗？",
                status="completed",
                metadata_json={
                    "final_response": {
                        "research_proposal": {
                            "query": "confirmed topic",
                            "proposal_id": "proposal",
                        }
                    }
                },
            )
        )
        db.commit()
        s = {**state(env), "user_input": reply}
        nodes = SupervisorNodes(db, {})
        await nodes.permission_guard(s)
        await nodes.bootstrap_context(s)
        assert s["research_authorized"] is authorized
        assert not s.get("approval_required")
        if authorized:
            from src.web_app.services.research_service import research_service

            seen = []

            async def research(db, state, arguments, **kwargs):
                seen.append(arguments["query"])
                return {"status": "completed"}

            monkeypatch.setattr(research_service, "research_for_supervisor", research)
            s["current_action"] = {
                "action": "deep_research",
                "action_id": "research",
                "arguments": {"query": "unconfirmed other topic"},
            }
            await execute_capability(nodes, s)
            assert seen == ["confirmed topic"]


def test_repeated_rag_actions_have_unambiguous_evidence_ids(env):
    from src.web_app.agent.runtime.policy import check_answer

    with env.factory() as db:
        nodes = SupervisorNodes(db, {})
        s = {**state(env), "runtime_budget": {"consecutive_failures": 0}}
        for action_id in ("first", "second"):
            observe(
                nodes,
                s,
                CapabilityResult(
                    action_id=action_id,
                    capability="rag",
                    status="ok",
                    summary="fact [E1]",
                    data={"answer": "fact [E1]"},
                    evidence=[{"evidence_id": "E1", "quote": action_id}],
                ),
            )
        first, second = s["observations"]
        assert first["evidence"][0]["evidence_id"] == "E1"
        assert second["evidence"][0]["evidence_id"] == "E2"
        assert second["data"]["answer"] == second["summary"] == "fact [E2]"
        assert not check_answer(nodes, s, "fact [E1] [E2]")
        assert "没有对应证据" in check_answer(nodes, s, "fact [E3]")


@pytest.mark.asyncio
async def test_invalid_decisions_terminate_without_keyword_planning(env, monkeypatch):
    from src.web_app.agent.runtime.graph_builder import build_graph
    from src.web_app.agent.runtime.state import SupervisorAction

    with env.factory() as db:
        nodes = SupervisorNodes(db, {})

        from src.web_app.agent.llm.native_turn import NativeProtocolError
        async def invalid(s, **kwargs):
            raise NativeProtocolError("invalid_tool_arguments")
        monkeypatch.setattr(nodes, "model_turn", invalid)
        result = await build_graph(nodes).ainvoke(state(env))
        assert result["status"] == "failed" and result["runtime_budget"]["steps"] == 3
        assert not result.get("tool_calls") and not result.get("research_result")
