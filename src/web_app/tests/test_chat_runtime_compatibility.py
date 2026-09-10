import asyncio
import json
from types import SimpleNamespace

import pytest
from langgraph.graph import StateGraph, END

from src.web_app.agent.llm.content import message_text
from src.web_app.agent.runtime.chat_control import ChatExecution, controlled_node, execution
from src.web_app.agent.runtime.nodes import RuntimeNodes


@pytest.mark.asyncio
async def test_compiled_graph_forwards_config_to_real_supervisor_method(monkeypatch):
    seen = []
    async def supervisor(state, config=None):
        seen.append(config)
        return state
    monkeypatch.setattr('src.web_app.agent.runtime.nodes.llm_supervisor_route_node', supervisor)
    nodes = RuntimeNodes(None, {'payload_option': 'kept'})
    graph = StateGraph(dict)
    graph.add_node('supervisor', controlled_node('llm_supervisor_route', nodes.llm_supervisor_route, None))
    async def plain(state):
        return {**state, 'plain_ran': True}
    graph.add_node('plain', controlled_node('plain', plain, None))
    graph.set_entry_point('supervisor')
    graph.add_edge('supervisor', 'plain')
    graph.add_edge('plain', END)
    token = execution.set(ChatExecution(1))
    try:
        result = await graph.compile().ainvoke({'run_id': 1, 'route_plan': {'intent': 'chat'}}, config={'configurable': {'probe': 'forwarded'}})
    finally:
        execution.reset(token)
    assert result['plain_ran']
    assert seen[0]['configurable']['probe'] == 'forwarded'
    assert seen[0]['configurable']['payload_option'] == 'kept'


@pytest.mark.asyncio
async def test_config_forwarding_keeps_cancellation_fence():
    async def forbidden(state, config=None):
        pytest.fail('Cancelled node executed')
    token = execution.set(ChatExecution(1, cancelled=True))
    try:
        with pytest.raises(asyncio.CancelledError):
            await controlled_node('llm_supervisor_route', forbidden, None)({'run_id': 1}, config={})
    finally:
        execution.reset(token)


@pytest.mark.parametrize('content, expected', [
    ('正文', '正文'),
    ([{'type': 'reasoning', 'summary': [{'text': 'private reasoning'}]}, {'type': 'text', 'text': '{"summary":"正文"}'}], '{"summary":"正文"}'),
    ([{'type': 'thinking', 'thinking': 'private'}, {'type': 'output_text', 'text': '回答'}, {'type': 'tool_use', 'name': 'tool'}], '回答'),
    ([{'type': 'reasoning', 'text': 'private'}], ''),
    ({'type': 'message', 'content': [{'type': 'output_text', 'text': '回答'}]}, '回答'),
])
def test_only_answer_text_is_extracted(content, expected):
    assert message_text(SimpleNamespace(content=content)) == expected


def test_summary_and_intent_ignore_reasoning_blocks(monkeypatch):
    from src.web_app.services.conversation_summary_service import _llm_call, _parse_json
    from src.web_app.agent.runtime.intent_llm import _message_content
    message = SimpleNamespace(content=[{'type': 'reasoning', 'summary': [{'text': 'private'}]}, {'type': 'text', 'text': '{"summary":"saved"}'}])
    monkeypatch.setattr('src.web_app.agent.llm.factory.get_chat_model', lambda *a, **k: SimpleNamespace(invoke=lambda *_: message))
    assert _parse_json(_llm_call('prompt')) == {'summary': 'saved'}
    assert json.loads(_message_content(message)) == {'summary': 'saved'}
