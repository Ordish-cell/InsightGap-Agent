import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import ts from 'typescript'
const source = readFileSync(new URL('../src/components/agent/liveProgress.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { collectLiveProgress } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
const event = (id, event_type, payload = {}, run_id = 2) => ({ id, event_seq: id, event_type, payload, run_id })

test('plain chat has no workflow or fabricated findings', () => {
  const result = collectLiveProgress([event(1, 'interaction_mode', { mode: 'chat' }), event(2, 'visible_thought', { text: 'template' })], 2)
  assert.equal(result.mode, 'chat')
  assert.deepEqual(result.blocks, [])
})
test('merge deltas, deduplicate replay, ignore other run and freeze terminal', () => {
  const a = event(3, 'progress_delta', { block_id: 'a', text: '真实' })
  const result = collectLiveProgress([event(1, 'interaction_mode', { mode: 'workflow' }),
    a, a, event(4, 'progress_delta', { block_id: 'a', text: '发现' }),
    event(5, 'progress_delta', { block_id: 'a', text: 'wrong' }, 9),
    event(6, 'run_interrupted'), event(7, 'progress_delta', { block_id: 'a', text: 'late' })], 2)
  assert.equal(result.blocks[0].text, '真实发现')
  assert.equal(result.stopped, true)
})
test('completed blocks replace accumulated text; retries are independent steps', () => {
  const events = [event(1, 'node_started', { step_id: 's1', display_name: '检索资料', status: 'running' }),
    event(2, 'node_failed', { step_id: 's1', display_name: '检索资料', status: 'failed' }),
    event(3, 'node_started', { step_id: 's2', display_name: '检索资料', status: 'running' }),
    event(4, 'progress_delta', { block_id: 'b', text: 'partial' }),
    event(5, 'progress_completed', { block_id: 'b', text: 'complete', references: [{ url: 'javascript:alert(1)' }, { url: 'https://example.test', title: '来源' }] })]
  const result = collectLiveProgress(events, 2)
  assert.equal(result.steps.length, 2)
  assert.equal(result.steps[0].status, 'failed')
  assert.equal(result.blocks[0].text, 'complete')
  assert.equal(result.blocks[0].references.length, 1)
  assert.deepEqual(collectLiveProgress(events, 2), result)
})

test('approval resume reopens progress, interruption never reopens the old run', () => {
  const tail = [event(3, 'run_resumed'), event(4, 'progress_completed', { block_id: 'after-approval', text: '工具结果已产生' })]
  const resumed = collectLiveProgress([event(1, 'interaction_mode', { mode: 'workflow' }), event(2, 'run_paused'), ...tail], 2)
  assert.equal(resumed.stopped, false)
  assert.equal(resumed.blocks.length, 1)
  const interrupted = collectLiveProgress([event(2, 'run_interrupted'), ...tail], 2)
  assert.equal(interrupted.stopped, true)
  assert.equal(interrupted.blocks.length, 0)
})
