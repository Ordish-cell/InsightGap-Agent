// Run against a local Vite server; all API calls are intercepted with synthetic data.
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const require = createRequire(import.meta.url)
const { chromium } = require('playwright')
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 980 } })
const errors = []
page.on('pageerror', error => errors.push(String(error)))
const catalog = [
  { key: 'custom', label: '自定义接口', protocol: 'openai_chat_completions', protocols: ['openai_chat_completions', 'openai_responses'], fields: [{ key: 'base_url', label: 'API 地址', kind: 'url', required: true }, { key: 'api_key', label: 'API Key', kind: 'secret' }, { key: 'custom_headers', label: '自定义 Headers', kind: 'secret_json' }], models: [] },
  { key: 'openai', label: 'OpenAI', protocol: 'openai_responses', protocols: ['openai_responses', 'openai_chat_completions'], fields: [{ key: 'api_key', label: 'API Key', kind: 'secret', required: true }, { key: 'base_url', label: 'API 地址', kind: 'url', default: 'https://api.openai.com/v1' }], models: [] },
]
const model = (id, connection_id, display_name, model_id) => ({ id, connection_id, display_name, model_id, enabled: true, source: 'manual', capabilities: { tools: true, streaming: true, structured_output: true } })
let connections = [
  { id: 1, provider: 'openai', protocol: 'openai_responses', display_name: 'OpenAI · 日常使用', status: 'active', last_test_status: 'passed', fields: { base_url: 'https://api.openai.com/v1' }, secrets: { api_key: { configured: true, masked: '••••1234' } }, models: [model(11, 1, '主力模型', 'text-main'), model(12, 1, '快速模型', 'text-fast')] },
  { id: 2, provider: 'custom', protocol: 'openai_chat_completions', display_name: '本地推理服务', status: 'active', last_test_status: 'passed', fields: { base_url: 'http://localhost:11434/v1' }, secrets: {}, models: [model(21, 2, '本地聊天', 'local-chat')] },
  { id: 3, provider: 'custom', protocol: 'openai_responses', display_name: '备用中转接口', status: 'draft', last_test_status: 'untested', fields: { base_url: 'https://relay.example.test/v1' }, secrets: {}, models: [] },
]
let defaultId = 11
let failTest = false
let testPayload
await page.addInitScript(() => localStorage.setItem('authToken', 'synthetic-browser-test'))
await page.route('**/api/v1/**', async route => {
  const request = route.request(), path = new URL(request.url()).pathname.replace('/api/v1', '')
  const payload = request.postDataJSON()
  let data = []
  if (path === '/auth/me') data = { id: 1, email: 'local@example.test', nickname: 'Local' }
  if (path === '/llm/catalog') data = catalog
  if (path === '/llm/connections') {
    if (request.method() === 'POST') {
      const item = { id: 4, ...payload, status: 'draft', last_test_status: 'untested', secrets: {}, models: [] }
      connections.push(item); data = item
    } else data = connections
  }
  if (path === '/llm/preferences') {
    if (request.method() === 'PATCH') defaultId = payload.default_model_config_id
    data = { default_model_config_id: defaultId }
  }
  if (path === '/llm/connections/test') {
    testPayload = payload
    if (failTest) return route.fulfill({ status: 400, json: { detail: 'authentication_failed' } })
    data = { status: 'ok', latency_ms: 85, persisted: true }
  }
  if (path.endsWith('/discover-models')) data = []
  await route.fulfill({ json: { success: true, data } })
})
const output = join(tmpdir(), 'insight-model-ui')
await mkdir(output, { recursive: true })
try {
  await page.goto(`${process.env.MODEL_UI_URL || 'http://127.0.0.1:5175'}/settings`)
  await page.getByRole('button', { name: '管理模型连接', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: '模型连接', exact: true })
  await dialog.getByRole('heading', { name: 'OpenAI · 日常使用' }).waitFor()
  await page.screenshot({ animations: 'disabled', path: join(output, 'models-desktop.png') })
  await dialog.getByRole('button', { name: /本地推理服务/ }).click()
  await dialog.getByRole('button', { name: '设为默认', exact: true }).click()
  await dialog.getByRole('status').filter({ hasText: '新会话默认使用' }).waitFor()
  assert.equal(defaultId, 21)
  await dialog.getByRole('button', { name: '测试生成', exact: true }).click()
  await dialog.getByRole('status').filter({ hasText: '85 ms' }).waitFor()
  assert.deepEqual(testPayload, { connection_id: 2, model_id: 'local-chat' })
  failTest = true
  await dialog.getByRole('button', { name: '测试生成', exact: true }).click()
  await dialog.getByRole('alert').filter({ hasText: '认证失败' }).waitFor()
  await dialog.getByRole('button', { name: '添加连接', exact: true }).click()
  await dialog.getByLabel('连接名称', { exact: true }).fill('临时连接')
  await dialog.getByLabel('API 地址').fill('http://localhost:1234/v1')
  await dialog.locator('summary').click()
  await dialog.getByLabel('自定义 Headers').fill('{invalid')
  await dialog.getByRole('button', { name: '保存连接', exact: true }).click()
  await dialog.getByRole('alert').filter({ hasText: '有效 JSON' }).waitFor()
  assert.equal(await dialog.getByLabel('连接名称', { exact: true }).inputValue(), '临时连接')
  await dialog.getByLabel('自定义 Headers').fill('{"X-Project":"local"}')
  await dialog.getByRole('button', { name: '保存连接', exact: true }).click()
  await dialog.getByRole('heading', { name: '临时连接', exact: true }).waitFor()
  assert.equal(connections.length, 4)
  await dialog.getByRole('button', { name: /OpenAI · 日常使用/ }).click()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ animations: 'disabled', path: join(output, 'models-mobile.png') })
  assert.equal(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), true)
  await page.keyboard.press('Escape')
  await dialog.waitFor({ state: 'detached' })
  assert.equal(await page.getByRole('button', { name: '管理模型连接', exact: true }).evaluate(el => el === document.activeElement), true)
  await page.setViewportSize({ width: 1440, height: 980 })
  await page.goto(`${process.env.MODEL_UI_URL || 'http://127.0.0.1:5175'}/agent`)
  await page.locator('.composer-model-button').click()
  const picker = page.getByRole('dialog', { name: '选择本次使用的模型' })
  await picker.getByRole('textbox', { name: '搜索供应商或模型' }).fill('OpenAI')
  await picker.getByRole('button', { name: /快速模型/ }).click()
  assert.match(await page.locator('.composer-model-button').textContent(), /快速模型/)
  await page.locator('.composer-model-button').click()
  await picker.getByRole('textbox', { name: '搜索供应商或模型' }).fill('本地')
  await picker.getByRole('button', { name: /本地聊天/ }).waitFor()
  await page.screenshot({ animations: 'disabled', path: join(output, 'model-picker.png') })
  await page.keyboard.press('Escape')
  await picker.waitFor({ state: 'detached' })
  assert.deepEqual(errors, [])
  console.log('PASS: provider switching, default selection, generation probe, readable failure, JSON editing, saved draft, narrow viewport, Escape/focus restoration')
  console.log(output)
} finally { await browser.close() }
