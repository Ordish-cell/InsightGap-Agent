import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/components/agent/chatStream.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { projectChatEvent, restoreChatMessage } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
const initial = () => ({ role: 'assistant', run_id: 1, message_id: 'm1', content: '', status: 'running' })
const event = (id, event_type, payload = {}) => ({ id, event_seq: id, run_id: 1, event_type, payload })

test('native text is immediate, progress is removed from final answer, final projection does not duplicate', () => {
  let message = initial()
  const apply = (e) => { message = projectChatEvent(message, e) }
  apply(event(1, 'agent_text_started', { text_id: 'a' }))
  apply(event(2, 'agent_text_delta', { text_id: 'a', text: 'Reading' }))
  assert.equal(message.content, 'Reading')
  apply(event(3, 'agent_text_completed', { text_id: 'a', role: 'progress', text: 'Reading' }))
  assert.equal(message.content, '')
  apply(event(4, 'agent_text_started', { text_id: 'b' }))
  apply(event(5, 'agent_text_delta', { text_id: 'b', text: 'Answer' }))
  apply(event(6, 'agent_text_completed', { text_id: 'b', role: 'final', text: 'Answer' }))
  apply(event(7, 'answer_delta', { text_id: 'b', text: 'Answer' }))
  assert.equal(message.content, 'Answer')
  apply(event(8, 'answer_completed', { text_id: 'b', answer: 'Answer + policy note' }))
  assert.equal(message.content, 'Answer + policy note')
})

test('history deduplicates events and preserves interrupted native text', () => {
  const events = [event(1, 'agent_text_started', { text_id: 'a' }), event(2, 'agent_text_delta', { text_id: 'a', text: 'Partial' })]
  const message = restoreChatMessage(initial(), [...events, events[1], event(3, 'run_interrupted', { answer: 'Partial' })])
  assert.equal(message.content, 'Partial')
  assert.equal(message.status, 'interrupted')
  assert.equal(projectChatEvent(message, event(4, 'agent_text_delta', { text_id: 'a', text: 'late' })).content, 'Partial')
})

test('other run or other message cannot update the current bubble', () => {
  const start = initial()
  assert.deepEqual(projectChatEvent(start, { ...event(1, 'agent_text_started', { text_id: 'a' }), run_id: 2 }), start)
  assert.deepEqual(projectChatEvent(start, event(1, 'agent_text_started', { text_id: 'a', message_id: 'other' })), start)
})


test('memory confirmation and save receipt suffix survive live projection and replay', () => {
  for (const suffix of ['是否保存为跨会话记忆？', '已保存跨会话记忆：称呼为「常」。', '称呼未确认保存成功。']) {
    const answer = `好的。\n\n${suffix}`
    const events = [
      event(1, 'agent_text_started', { text_id: 'fact' }),
      event(2, 'agent_text_delta', { text_id: 'fact', text: '好的。' }),
      event(3, 'agent_text_completed', { text_id: 'fact', role: 'final', text: '好的。' }),
      event(4, 'answer_completed', { text_id: 'fact', answer }),
      event(5, 'run_completed', { answer }),
    ]
    const live = events.reduce(projectChatEvent, initial())
    assert.equal(live.content, answer)
    assert.equal(restoreChatMessage(initial(), events).content, answer)
    assert.equal(restoreChatMessage(live, events).content, answer)
  }
})
