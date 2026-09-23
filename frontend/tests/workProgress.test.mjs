import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/components/agent/workProgress.ts', import.meta.url), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { projectWorkProgress: project } = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`)
const message = { run_id: 1, message_id: 'm', role: 'assistant', status: 'streaming' }
const e = (id, event_type, payload = {}, node_name = '') => ({ id, run_id: 1, event_type, payload, node_name })
const start = e(1, 'agent_text_started', { text_id: 'a' })
const delta = e(2, 'agent_text_delta', { text_id: 'a', text: '先读取文档。' })
const progress = e(3, 'agent_text_completed', { text_id: 'a', role: 'progress', text: '先读取文档。' })
test('partial text stays in the same ordered block through classification, tool and final answer', () => {
  const before = project(message, [start, delta], '')
  assert.equal(before.blocks[0].text, '先读取文档。')
  const events = [start, delta, progress, e(4, 'node_started', { step_id: 'read', display_name: '读取文档', status: 'running' }, 'document_read'), e(5, 'node_completed', { step_id: 'read', display_name: '读取文档', status: 'completed' }, 'document_read'), e(6, 'agent_text_started', { text_id: 'b' }), e(7, 'agent_text_delta', { text_id: 'b', text: '文件分为三个阶段。' }), e(8, 'agent_text_completed', { text_id: 'b', role: 'final', text: '文件分为三个阶段。' })]
  const after = project(message, events, '')
  assert.equal(after.blocks[0].id, before.blocks[0].id)
  assert.deepEqual(after.blocks.map(b => b.kind), ['text', 'tool', 'text'])
  assert.equal(after.blocks[1].status, 'completed')
  const answer = '文件分为三个阶段。\n\n是否保存为跨会话记忆？'
  const final = project({ ...message, status: 'completed' }, [...events, e(9, 'answer_completed', { answer }), e(10, 'run_completed', { answer })], answer)
  assert.equal(final.blocks[0].text, before.blocks[0].text)
  assert.equal(final.blocks.at(-1).text, answer)
  assert.equal(final.blocks.length, 3)
})
test('duplicate, out-of-order, wrong message and wrong run events cannot corrupt text', () => {
  const events = [delta, start, delta, { ...delta, id: 3, run_id: 2 }, e(4, 'agent_text_delta', { text_id: 'a', message_id: 'other', text: 'wrong' })]
  assert.equal(project(message, events, '').blocks[0].text, '先读取文档。')
})
test('stop retains partial text, rejects late text and never marks running tools successful', () => {
  const events = [start, delta, e(3, 'tool_call_started', { tool_call_id: 't', tool_name: 'web.search' }), e(4, 'run_interrupted'), e(5, 'agent_text_delta', { text_id: 'a', text: 'late' }), e(6, 'run_resumed')]
  const trace = project({ ...message, status: 'interrupted' }, events, '先读取文档。')
  assert.equal(trace.blocks[0].text, '先读取文档。')
  assert.equal(trace.blocks[1].status, 'cancelled')
  assert.equal(trace.active, false)
})
test('approval resume continues; no preamble produces only actual activity', () => {
  const events = [start, e(2, 'agent_text_completed', { text_id: 'a', role: 'progress', text: '' }), e(3, 'run_paused'), e(4, 'agent_text_delta', { text_id: 'a', text: 'late' }), e(5, 'run_resumed'), e(6, 'tool_call_started', { tool_call_id: 't', tool_name: 'memory.save' }), e(7, 'tool_call_failed', { tool_call_id: 't', tool_name: 'memory.save' })]
  assert.deepEqual(project(message, events, '').blocks.map(b => [b.kind, b.status]), [['tool', 'failed']])
})
test('concrete tool replaces its generic lifecycle without a duplicate completion row', () => {
  const events = [start, progress, e(4, 'node_started', { step_id: 'x', display_name: '执行工具', status: 'running' }, 'tool_runtime'), e(5, 'tool_call_started', { tool_call_id: 't', tool_name: 'web.search' }), e(6, 'tool_call_completed', { tool_call_id: 't', tool_name: 'web.search' }), e(7, 'node_completed', { step_id: 'x', display_name: '执行工具', status: 'completed' }, 'tool_runtime')]
  assert.equal(project(message, events, '').blocks.filter(b => b.kind === 'tool').length, 1)
})

test('protocol repair and generic lifecycle stay in execution details', () => {
  const events = [start, progress, e(4, 'node_started', { step_id: 'repair', display_name: '处理请求', status: 'running' }, 'capability'), e(5, 'node_completed', { step_id: 'repair', display_name: '处理请求', status: 'completed' }, 'capability')]
  const trace = project(message, events, '')
  assert.deepEqual(trace.blocks.map(b => b.kind), ['text'])
  assert.equal(trace.steps[0].text, '处理请求')
})
