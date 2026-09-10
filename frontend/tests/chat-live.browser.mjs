// Full chat UI + real chunked HTTP SSE, backed by a synthetic in-memory ledger.
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
const { chromium } = createRequire(import.meta.url)('playwright')
const runs = new Map(), ledgers = new Map(), streams = new Map()
let nextRun = 0, seq = 0
const messages = []
const connection = { id: 1, provider: 'custom', protocol: 'openai_chat_completions', display_name: '隔离测试模型', status: 'active', last_test_status: 'passed', fields: {}, secrets: {}, models: [{ id: 11, connection_id: 1, model_id: 'fake', display_name: '测试模型', enabled: true, capabilities: { streaming: true } }] }
function emit(runId, event_type, payload = {}) {
  const event = { id: ++seq, event_seq: seq, run_id: runId, event_type, payload, created_at: new Date().toISOString() }
  ledgers.get(runId).push(event)
  streams.get(runId)?.write(`id: ${seq}\nevent: ${event_type}\ndata: ${JSON.stringify(event)}\n\n`)
  return event
}
const server = createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, Last-Event-ID')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  if (req.method === 'OPTIONS') { res.end(); return }
  const url = new URL(req.url, 'http://localhost'), path = url.pathname.replace('/api/v1', '')
  const chunks = []; for await (const chunk of req) chunks.push(chunk)
  const body = chunks.length ? JSON.parse(Buffer.concat(chunks)) : {}
  let data = []
  if (path === '/auth/me') data = { id: 1, email: 'isolated@example.test' }
  if (path === '/llm/connections') data = [connection]
  if (path === '/llm/preferences') data = { default_model_config_id: 11 }
  if (path === '/llm/catalog') data = [{ key: 'custom', label: '自定义', fields: [] }]
  if (path === '/agent/conversations') data = { items: nextRun ? [{ conversation_id: 'isolated', title: '隔离测试' }] : [] }
  if (path === '/agent/conversations/isolated') data = { conversation_id: 'isolated', messages, metadata: { model_config_id: 11 } }
  if (path === '/agent/runs/start') {
    const runId = ++nextRun
    const user = { role: 'user', message_id: `user-${runId}`, run_id: runId, content: body.user_input, status: 'completed', conversation_id: 'isolated' }
    const assistant = { role: 'assistant', message_id: `assistant-${runId}`, run_id: runId, content: '', status: 'thinking', conversation_id: 'isolated', metadata: { interaction_version: 2 } }
    messages.push(user, assistant)
    data = { run_id: runId, conversation_id: 'isolated', status: 'running', can_interrupt: true, can_steer: true, user_message: user, assistant_message: assistant, last_event_seq: ++seq, interaction_version: 2 }
    runs.set(runId, data); ledgers.set(runId, [])
  }
  const match = path.match(/^\/agent\/runs\/(\d+)(.*)$/)
  if (match) {
    const id = Number(match[1]), suffix = match[2]
    if (suffix === '/events/stream') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' }); res.flushHeaders()
      streams.set(id, res)
      for (const e of ledgers.get(id) || []) if (e.id > Number(url.searchParams.get('after_seq') || 0)) res.write(`id: ${e.id}\ndata: ${JSON.stringify(e)}\n\n`)
      return
    }
    if (suffix === '/events') data = { run_id: id, events: ledgers.get(id), next_seq: seq, until_seq: seq, has_more: false }
    else data = runs.get(id)
  }
  res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({ success: true, data }))
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
const api = `http://127.0.0.1:${server.address().port}/api/v1`
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } })
const errors = []; page.on('pageerror', error => errors.push(String(error)))
await page.addInitScript(api => { localStorage.setItem('authToken', 'isolated'); localStorage.setItem('apiBaseUrl', api) }, api)
const output = join(tmpdir(), 'insight-live-progress'); await mkdir(output, { recursive: true })
try {
  await page.goto(`${process.env.CHAT_UI_URL || 'http://127.0.0.1:5173'}/`)
  await page.locator('.composer-model-button').filter({ hasText: '测试模型' }).waitFor()
  await page.locator('textarea').fill('唉，我好累'); await page.locator('textarea').press('Enter')
  await page.getByRole('status').filter({ hasText: '正在生成回复' }).waitFor()
  assert.equal(await page.locator('.activity-timeline').count(), 0)
  while (!streams.has(1)) await new Promise(r => setTimeout(r, 10))
  emit(1, 'interaction_mode', { mode: 'chat', interaction_version: 2 })
  emit(1, 'answer_delta', { message_id: 'assistant-1', text: '听起来你很累，先歇一会儿吧。' })
  await page.getByText('听起来你很累，先歇一会儿吧。', { exact: true }).waitFor()
  const finish = id => {
    const run = runs.get(id); run.status = 'completed'; run.can_interrupt = false; run.can_steer = false
    run.assistant_message.status = 'completed'; run.assistant_message.content = id === 1 ? '听起来你很累，先歇一会儿吧。' : '主要等待来自意图识别。'
    run.answer = run.assistant_message.content
    emit(id, 'answer_completed', { answer: run.answer, message_id: run.assistant_message.message_id })
    emit(id, 'run_completed', { status: 'completed', answer: run.answer, response: run }); streams.get(id).end()
  }
  finish(1)
  await page.screenshot({ path: join(output, 'full-chat.png') })
  await page.locator('textarea').fill('帮我分析项目为什么这么慢'); await page.locator('textarea').press('Enter')
  while (!streams.has(2)) await new Promise(r => setTimeout(r, 10))
  emit(2, 'interaction_mode', { mode: 'workflow', interaction_version: 2 })
  emit(2, 'node_started', { step_id: 'inspect', status: 'running', display_name: '检查模型调用' })
  await page.getByRole('status').filter({ hasText: '检查模型调用' }).waitFor()
  const samples = []
  for (let i = 0; i < 30; i++) {
    const started = Date.now(), text = `意图识别等待 42 秒，回答生成尚未开始。记录 ${i + 1}`
    emit(2, 'progress_completed', { step_id: 'inspect', block_id: 'finding', text })
    await page.getByText(text, { exact: true }).waitFor()
    await page.evaluate(() => new Promise(r => requestAnimationFrame(r)))
    samples.push(Date.now() - started)
  }
  assert.equal(await page.locator('.live-progress-details').getAttribute('open'), null)
  await page.screenshot({ path: join(output, 'full-workflow.png') })
  finish(2)
  await page.getByText('主要等待来自意图识别。', { exact: true }).waitFor()
  await page.reload()
  await page.getByText('主要等待来自意图识别。', { exact: true }).waitFor()
  assert.equal(await page.locator('.live-progress-finding').count(), 1)
  assert.deepEqual(errors, [])
  const p95 = samples.sort((a, b) => a - b)[28]
  assert.ok(p95 <= 300, `SSE-to-paint p95 ${p95} ms`)
  console.log(JSON.stringify({ passed: true, samples: 30, synthetic_ledger_sse_to_paint_p95_ms: p95, screenshots: output }))
} catch (error) {
  console.error('runs', nextRun, 'errors', errors, 'body', (await page.locator('body').innerText()).slice(-1600))
  await page.screenshot({ path: join(output, 'full-failure.png') })
  throw error
} finally { await browser.close(); server.closeAllConnections(); server.close() }
