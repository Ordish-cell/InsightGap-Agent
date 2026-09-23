// Attachment, stop and replay UI regression with a local synthetic API. No real model calls.
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
let stops = 0, uploads = 0
// Default to the existing compatibility stream for the UI-only branch.
// CHAT_PROTOCOL=native also exercises the separately maintained native message-ID fix.
const nativeProtocol = process.env.CHAT_PROTOCOL === 'native'
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
  const body = chunks.length && req.headers['content-type']?.includes('application/json') ? JSON.parse(Buffer.concat(chunks)) : {}
  let data = []
  if (path === '/documents/chat-upload') { uploads++; data = { document_id: 99, kind: 'document', filename: 'notes.txt', chunks_count: 1, status: 'ready', ingest_status: 'completed' } }
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
    else if (suffix === '/interrupt') {
      stops++
      const run = runs.get(id)
      run.status = 'interrupted'; run.can_interrupt = false; run.can_steer = false
      run.assistant_message.status = 'interrupted'; run.assistant_message.content = '已生成的部分回复。'
      run.answer = run.assistant_message.content
      emit(id, 'run_interrupted', { status: 'interrupted', response: run })
      streams.get(id)?.end()
      data = { client_command_id: body.client_command_id, run_id: id, successor_run_id: null, kind: 'interrupt', status: 'applied' }
    } else data = runs.get(id)
  }
  res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({ success: true, data }))
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
const api = `http://127.0.0.1:${server.address().port}/api/v1`
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } })
const errors = []; page.on('pageerror', error => errors.push(String(error)))
await page.addInitScript(api => { localStorage.setItem('authToken', 'isolated'); localStorage.setItem('apiBaseUrl', api) }, api)
const output = join(tmpdir(), 'insightgap-chat-redesign'); await mkdir(output, { recursive: true })
try {
  await page.goto(`${process.env.CHAT_UI_URL || 'http://127.0.0.1:5175'}/`)
  await page.locator('.composer-model-button').filter({ hasText: '测试模型' }).waitFor()
  await page.locator('input[type=file]').setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('isolated attachment') })
  await page.getByText('已上传', { exact: true }).waitFor()
  assert.equal(uploads, 1)
  await page.screenshot({ path: join(output, 'attachment.png') })
  await page.getByRole('button', { name: '移除附件', exact: true }).click()
  assert.equal(await page.locator('.composer-attachment-card').count(), 0)
  await page.locator('textarea').fill('测试停止与回放')
  await page.locator('textarea').press('Enter')
  await page.getByRole('button', { name: '停止生成' }).waitFor()
  for (let i = 0; i < 100 && !streams.has(1); i++) await new Promise(resolve => setTimeout(resolve, 20))
  assert.ok(streams.has(1))
  emit(1, 'interaction_mode', { mode: 'chat', interaction_version: 2 })
  if (nativeProtocol) {
    emit(1, 'agent_text_started', { message_id: 'assistant-1', text_id: 'native-1' })
    emit(1, 'agent_text_delta', { message_id: 'assistant-1', text_id: 'native-1', text: '已生成的部分回复。' })
  } else {
    emit(1, 'answer_delta', { text: '已生成的部分回复。' })
  }
  await page.getByText('已生成的部分回复。', { exact: true }).waitFor()
  // Folding the feed must not replace the chat DOM or its scroll position.
  const marker = await page.locator('textarea').evaluate(el => { el.dataset.continuity = 'same'; return el.dataset.continuity })
  const scroll = await page.locator('.chat-scroll').evaluate(el => el.scrollTop)
  await page.getByRole('button', { name: /今日精选/ }).click()
  await page.getByRole('button', { name: /今日精选/ }).click()
  assert.equal(await page.locator('textarea').getAttribute('data-continuity'), marker)
  assert.equal(await page.locator('.chat-scroll').evaluate(el => el.scrollTop), scroll)
  await page.getByRole('button', { name: '停止生成' }).click()
  await page.getByText('已中断 · 已保留部分回复', { exact: true }).waitFor()
  assert.equal(stops, 1)
  assert.equal(await page.getByRole('button', { name: '停止生成' }).count(), 0)
  await page.reload()
  await page.getByText('已生成的部分回复。', { exact: true }).waitFor()
  assert.equal(await page.getByText('已生成的部分回复。', { exact: true }).count(), 1)
  await page.screenshot({ path: join(output, 'chat-replay.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(300)
  await page.screenshot({ path: join(output, 'chat-mobile.png') })
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ passed: true, protocol: nativeProtocol ? 'native' : 'compatibility', attachments: uploads, stops, preservedChatDOM: true, replay: true, screenshots: output }))
} finally { await browser.close(); server.closeAllConnections(); server.close() }
