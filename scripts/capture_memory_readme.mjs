// Capture the current UI with synthetic API responses; no real account or model calls.
// Requires Playwright and the frontend dev server (WORKBENCH_URL, default :5173).
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
const { chromium } = createRequire(import.meta.url)('playwright')
const output = new URL('../images/workbench/', import.meta.url)
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
const errors = []
page.on('pageerror', error => errors.push(String(error)))
let profile = { use_memory: true, generate_memory: false }
const common = { memory_type: 'semantic', importance: .8, effective_importance: .8, confidence: 1, evidence_count: 1, last_seen_at: '2026-09-30T09:00:00Z' }
const memories = [
  { ...common, id: 12, content: '用户偏好简体中文，回答先给结论，再说明依据。', category: 'answer_preference', status: 'active', scope: 'user', scope_id: '', metadata: { source_quote: '以后用简体中文回答，先给结论，再解释依据。', source_message_id: 101, run_id: 51 } },
  { ...common, id: 15, content: '本项目数据库方案已改为 PostgreSQL，继续保留原有数据。', category: 'tech_stack', status: 'active', scope: 'conversation', scope_id: 'demo-memory', metadata: { source_quote: '这个项目换成 PostgreSQL，但保留原有数据。', source_message_id: 108, run_id: 54, supersedes: 14 } },
  { ...common, id: 18, confidence: .65, content: '当前项目可能需要切换检索方案，尚未明确是否替代现有实现。', category: 'project_goal', status: 'pending', scope: 'conversation', scope_id: 'demo-memory', metadata: { source_quote: '要不试试另一种检索方案？先比较效果再决定。', source_message_id: 112, run_id: 56 } },
]
await page.addInitScript(() => {
  localStorage.setItem('authToken', 'readme-demo-only')
  localStorage.setItem('apiBaseUrl', 'http://127.0.0.1:8000/api/v1')
})
await page.route('**/api/v1/**', async route => {
  const request = route.request(), url = new URL(request.url()), path = url.pathname.replace('/api/v1', '')
  let data = []
  if (path === '/auth/me') data = { id: 1, nickname: '演示用户', email: 'demo@example.test' }
  if (path === '/profile/me') {
    if (request.method() === 'PUT') profile = { ...profile, ...request.postDataJSON() }
    data = profile
  }
  if (path === '/agent/conversations') data = { items: [], total: 0 }
  if (path === '/llm/preferences') data = { default_model_config_id: null }
  if (path === '/memory/summary') data = { semantic_count: 2, episodic_count: 0 }
  if (path === '/memory/growth-profile') data = { semantic_count: 2, episodic_count: 0, categories: [], recent_episodic: [] }
  if (path === '/memory/long-term' || path === '/memory/long-term/search') {
    const status = url.searchParams.get('status'), scope = url.searchParams.get('scope')
    const items = memories.filter(m => (!status || status === 'all' || m.status === status) && (!scope || m.scope === scope))
    data = { items, total: items.length, page: 1, page_size: 20 }
  }
  await route.fulfill({ json: { success: true, data } })
})
try {
  await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/memory`)
  const generation = page.getByRole('checkbox', { name: '自动整理后续对话' })
  await generation.waitFor({ state: 'visible' })
  await generation.click()
  await page.waitForFunction(() => {
    const input = document.querySelectorAll('.memory-controls input')[1]
    return input?.checked && !input.disabled
  })
  assert.equal(profile.generate_memory, true)
  await page.getByText('本项目数据库方案已改为 PostgreSQL，继续保留原有数据。', { exact: true }).waitFor()
  await page.evaluate(() => document.fonts.ready)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.screenshot({ path: fileURLToPath(new URL('memory.png', output)), fullPage: true, animations: 'disabled' })
  const status = page.locator('.filter-bar select').nth(2)
  await status.selectOption('pending')
  await page.getByRole('button', { name: '确认记忆', exact: true }).waitFor()
  await page.getByRole('combobox', { name: '记忆作用域' }).selectOption('conversation')
  await page.getByText('要不试试另一种检索方案？先比较效果再决定。', { exact: true }).waitFor()
  await page.screenshot({ path: fileURLToPath(new URL('memory-pending.png', output)), fullPage: true, animations: 'disabled' })
  await page.goto(process.env.WORKBENCH_URL || 'http://127.0.0.1:5173')
  await page.locator('textarea').waitFor()
  assert.equal(await page.locator('.composer-dock .memory-controls').count(), 0)
  assert.deepEqual(errors, [])
  console.log('Updated memory.png and memory-pending.png; chat composer has no memory controls.')
} finally {
  await browser.close()
}
