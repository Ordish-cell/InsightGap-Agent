import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import ts from 'typescript'

const source = readFileSync(new URL('../src/components/agent/chatStream.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { projectChatEvent, restoreChatMessage } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
const message = { role: 'assistant', run_id: 2, message_id: 'm2', status: 'thinking', content: '' }
const event = (id, event_type, payload, run_id = 2) => ({ id, event_seq: id, event_type, payload, run_id })

test('late run and wrong message output never reaches the successor', () => {
  assert.equal(projectChatEvent(message, event(1, 'answer_delta', { text: 'old' }, 1)), message)
  assert.equal(projectChatEvent(message, event(2, 'answer_delta', { text: 'old', message_id: 'm1' })), message)
})

test('refresh reconstructs unfinished text once and applies interruption', () => {
  const first = event(1, 'answer_delta', { text: '部分' })
  const replay = [first, first, event(2, 'answer_delta', { text: '回复' }), event(3, 'run_interrupted', { answer: '部分回复' })]
  const restored = restoreChatMessage(message, replay)
  assert.equal(restored.content, '部分回复')
  assert.equal(restored.status, 'interrupted')
  assert.equal(projectChatEvent(restored, event(4, 'answer_delta', { text: '迟到' })), restored)
  assert.equal(projectChatEvent(restored, event(5, 'answer_completed', { answer: '旧结论' })), restored)
  assert.equal(projectChatEvent(restored, event(6, 'run_completed', { answer: '旧结论' })), restored)
  assert.equal(restoreChatMessage(restored, replay), restored)
})

test('queued successor accepts output; answer end is not run completion', () => {
  let next = projectChatEvent({ ...message, status: 'queued' }, event(1, 'answer_delta', { text: '接续' }))
  next = projectChatEvent(next, event(2, 'answer_completed', { answer: '接续内容' }))
  assert.equal(next.status, 'streaming')
  next = projectChatEvent(next, event(3, 'run_completed', { answer: '接续内容' }))
  assert.equal(next.status, 'completed')
})

test('failed control/run preserves the saved partial reply', () => {
  const failed = projectChatEvent({ ...message, content: '部分' }, event(1, 'run_failed', { error: 'timeout' }))
  assert.equal(failed.content, '部分')
  assert.equal(failed.error_message, 'timeout')
})
