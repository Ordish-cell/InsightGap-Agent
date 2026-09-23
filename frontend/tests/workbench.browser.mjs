// Local-only UI regression. Every API request is fulfilled with synthetic fixtures.
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const { chromium } = createRequire(import.meta.url)('playwright')
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
const output = join(tmpdir(), 'insightgap-workbench-redesign')
await mkdir(output, { recursive: true })
const base = process.env.WORKBENCH_URL || 'http://127.0.0.1:5175'
const errors = [], writes = []
let mode = 'normal', failAction = false
page.on('pageerror', error => errors.push(String(error)))
await page.addInitScript(() => localStorage.setItem('authToken', 'synthetic-ui-test'))
const cards = ['explicit_related', 'adjacent_domain', 'far_domain'].map((bucket, i) => ({
  id: i + 1, title: ['让检索结果更可解释：从引用到证据链', '小团队如何建立可持续的信息研究流程', '从城市公共空间中发现协作设计的启发'][i],
  summary: '一份关于信息获取、证据判断与实践方法的观察。结合具体案例，梳理可验证的结论与仍待探索的问题。',
  information_gap: '从单条信息走向可追溯的判断，需要保留来源与上下文。', why_you: '画像中关注「信息检索」，该信号也涉及信息检索。',
  final_score: .86 - i / 10, exposure_bucket: bucket, relation_type: bucket, domain: 'example.test', source_type: 'web', source_url: 'https://example.test',
  evidence: [{ title: '研究方法与实践记录', url: 'https://example.test', snippet: '提供原始资料与可复现的方法。' }], suggested_actions: ['阅读原始材料，核对研究边界。'],
}))
const runs = [{ id: 'r1', query: '小团队如何建立可靠的信息研究流程？', status: 'completed', artifact_id: 1, summary: '以证据为基础，保留判断过程。', markdown_report: '# 研究结论\n\n研究应从明确的问题开始，而后逐步收集证据。\n\n## 实施建议\n\n- 保留信息来源\n- 定期核对假设\n\n| 阶段 | 产出 |\n| --- | --- |\n| 检索 | 证据清单 |\n| 综合 | 可追溯报告 |', metadata: { engine: 'open_deep_research' } }]
let memories = [{ id: 1, content: '用户希望被称呼为小常。', memory_type: 'semantic', category: 'preferred_name', status: 'active', importance: .8, effective_importance: .8, confidence: 1 }]
let skills = [{ id: 1, name: '资料核验与报告整理', description: '核对来源，整合关键信息，形成有依据的报告。', trigger_text: '当需要整理研究材料时', status: 'draft', safety_level: 'L1' }, { id: 2, name: '研究摘要', description: '将长文转化为可阅读的摘要。', status: 'approved', safety_level: 'L1' }]
let approvals = [{ id: 1, title: '保存整理后的研究材料', description: '核对以下操作后决定是否继续。', status: 'pending', payload: { permission_level: 'L3' } }, { id: 2, title: '已处理的操作', status: 'approved', payload: { permission_level: 'L3' } }, { id: 3, title: '受限操作', status: 'pending', payload: { permission_level: 'L4' } }]
await page.route('**/api/v1/**', async route => {
  const request = route.request(), url = new URL(request.url()), path = url.pathname.replace('/api/v1', ''), method = request.method()
  if (method !== 'GET') { writes.push(path); await new Promise(resolve => setTimeout(resolve, 180)); if (failAction) return route.fulfill({ status: 500, json: { detail: '测试操作失败，请重试。' } }) }
  if (mode === 'error' && path !== '/auth/me') return route.fulfill({ status: 503, json: { detail: '测试服务暂时不可用' } })
  if (mode === 'loading' && path !== '/auth/me') await new Promise(resolve => setTimeout(resolve, 800))
  let data = []
  if (path === '/auth/me') data = { id: 1, nickname: '小常', email: 'local@example.test' }
  if (path === '/feed/home') data = { cards, is_complete: true }
  if (path === '/feed/cards') data = mode === 'empty' ? [] : mode === 'long' ? cards.map(card => ({ ...card, title: card.title.repeat(8), summary: 'A'.repeat(300), why_you: card.why_you.repeat(8) })) : cards
  if (/^\/feed\/cards\/\d+$/.test(path)) data = cards[0]
  if (path.endsWith('/research')) data = { id: 'r1' }
  if (path === '/feed/stats') data = { cards_count: 3, saved_count: 1, hidden_count: 0, average_final_score: .76 }
  if (path === '/research/runs') data = method === 'GET' ? (mode === 'empty' ? [] : runs) : runs[0]
  if (path === '/research/runs/r1') data = runs[0]
  if (path === '/artifacts') data = mode === 'empty' ? [] : [{ id: 1, title: '信息研究流程 · 实践报告', artifact_type: 'markdown_report' }, { id: 2, title: '来源清单', artifact_type: 'text' }]
  if (path.startsWith('/artifacts/')) data = { id: Number(path.split('/').at(-1)), title: path.endsWith('1') ? '信息研究流程 · 实践报告' : '来源清单', artifact_type: path.endsWith('1') ? 'markdown_report' : 'text', content: path.endsWith('1') ? runs[0].markdown_report : '这是第二份成果。' }
  if (path === '/memory/summary') data = { semantic_count: memories.filter(m => m.status === 'active').length, episodic_count: 0 }
  if (path === '/memory/growth-profile') data = { semantic_count: memories.filter(m => m.status === 'active').length, episodic_count: 0, categories: [{ category: 'preferred_name', label: '称呼', count: 1, memories }], recent_episodic: [] }
  if (path === '/memory/long-term' || path === '/memory/long-term/search') { const items = mode === 'empty' ? [] : memories.filter(m => url.searchParams.get('status') === 'all' || m.status === 'active'); data = { items, total: items.length, page: 1 } }
  if (path.endsWith('/archive')) memories[0].status = 'archived'
  if (path.endsWith('/restore')) memories[0].status = 'active'
  if (path === '/memory/1' && method === 'DELETE') memories = []
  if (path === '/memory/consolidate') data = { reason: 'not_enough_memories' }
  if (path === '/skills') data = mode === 'empty' ? [] : skills
  if (/^\/skills\/\d+\//.test(path)) { const item = skills.find(s => s.id === Number(path.split('/')[2])); item.status = path.endsWith('/approve') ? 'approved' : 'disabled'; data = item }
  if (path === '/approvals') data = mode === 'empty' ? [] : approvals
  if (/^\/approvals\/\d+\//.test(path)) { const item = approvals.find(s => s.id === Number(path.split('/')[2])); item.status = path.endsWith('/approve') ? 'approved' : 'rejected'; data = item }
  if (path === '/mcp/tools') data = mode === 'empty' ? [] : [{ name: 'search_documents', description: '检索已保存的资料，返回相关片段与引用。', safety_level: 'L0' }]
  if (path === '/mcp/tool-calls') data = mode === 'empty' ? [] : [{ id: 1, tool_name: 'search_documents', status: 'completed', safety_level: 'L0', input: { query: '研究方法' }, output: { results: 3 } }]
  if (path === '/llm/preferences') data = { default_model_config_id: null }
  if (path === '/agent/conversations') data = { items: [], total: 0 }
  await route.fulfill({ json: { success: true, data } })
})
const routes = ['/', '/feed', '/feed/1', '/research', '/research/r1', '/artifacts', '/memory', '/skills', '/approvals', '/mcp', '/settings', '/profile', '/agent']
async function visit(path) { await page.goto(base + path); await page.waitForTimeout(400) }
async function screenshot(name) { await page.screenshot({ path: join(output, name + '.png'), fullPage: true }) }
try {
  for (const width of [390, 768, 1440, 1920]) {
    await page.setViewportSize({ width, height: width < 800 ? 844 : 1000 })
    for (const path of routes) {
      await visit(path)
      const overflow = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, width: innerWidth }))
      assert.ok(overflow.scroll <= width + 1, `${path} overflows at ${width}: ${overflow.scroll}`)
      await screenshot(`${width}-${path.replaceAll('/', '_') || 'home'}`)
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 })
  await visit('/')
  assert.equal(await page.locator('.home-feed-disclosure').getAttribute('inert'), '')
  await page.locator('textarea').fill('保留输入内容')
  await page.getByRole('button', { name: /今日精选/ }).click()
  assert.equal(await page.locator('textarea').inputValue(), '保留输入内容')
  await screenshot('home-expanded')
  await page.getByRole('button', { name: /今日精选/ }).click()
  await page.locator('.primary-navigation').getByRole('link', { name: '信息流', exact: true }).click()
  const navAnimations = await page.locator('.nav-active-indicator').evaluate(el => el.getAnimations().length)
  assert.ok(navAnimations > 0, 'navigation indicator should animate')
  await screenshot('motion-navigation-midpoint')
  await page.waitForTimeout(240)
  await screenshot('motion-navigation-settled')
  await page.getByRole('button', { name: '反馈', exact: true }).first().click()
  await page.getByRole('button', { name: '有用', exact: true }).click()
  await page.getByRole('status').filter({ hasText: '已反馈' }).waitFor()
  failAction = true
  await page.getByRole('button', { name: '反馈', exact: true }).first().click()
  await page.getByRole('button', { name: '保存', exact: true }).click()
  await page.getByRole('alert').filter({ hasText: '测试操作失败' }).waitFor()
  failAction = false
  await visit('/research')
  assert.equal(await page.getByRole('button', { name: '创建研究' }).isDisabled(), true)
  await page.locator('textarea').fill('合成数据研究，不访问模型')
  await page.getByRole('button', { name: '创建研究' }).dblclick()
  await page.waitForURL('**/research/r1')
  assert.equal(writes.filter(path => path === '/research/runs').length, 1)
  await visit('/artifacts')
  await page.locator('.artifact-reader .md-h1').waitFor()
  await page.getByRole('button', { name: /来源清单/ }).click()
  await page.getByText('这是第二份成果。', { exact: true }).waitFor()
  await visit('/skills')
  await page.getByRole('button', { name: '批准', exact: true }).click()
  await page.getByRole('status').filter({ hasText: '已启用' }).waitFor()
  assert.equal(await page.getByRole('button', { name: '批准', exact: true }).count(), 0)
  await visit('/approvals')
  assert.equal(await page.locator('.approval-card').nth(1).getByRole('button').count(), 0)
  assert.equal(await page.locator('.approval-card').nth(2).getByRole('button', { name: '批准' }).isDisabled(), true)
  await page.getByRole('button', { name: '批准', exact: true }).first().click()
  await page.getByRole('status').filter({ hasText: '批准已提交' }).waitFor()
  await visit('/memory')
  await page.getByRole('button', { name: '归档', exact: true }).click()
  await page.getByText('暂无长期记忆', { exact: true }).waitFor()
  await page.locator('select').nth(2).selectOption('all')
  await page.getByRole('button', { name: '恢复', exact: true }).click()
  await page.getByRole('status').filter({ hasText: '已恢复' }).waitFor()
  await page.getByRole('button', { name: '删除', exact: true }).click()
  await page.getByRole('alertdialog').waitFor()
  await screenshot('memory-confirm')
  await page.keyboard.press('Escape')
  await page.waitForTimeout(200)
  assert.equal(await page.getByRole('alertdialog').count(), 0)
  await page.getByRole('tab', { name: '长期记忆', exact: true }).focus()
  await page.keyboard.press('ArrowRight')
  assert.equal(await page.getByRole('tab', { name: '长期设定' }).getAttribute('aria-selected'), 'true')
  await visit('/settings')
  await page.getByRole('button', { name: '保存服务地址' }).click()
  await page.getByRole('status').filter({ hasText: '已保存' }).waitFor()
  await page.setViewportSize({ width: 390, height: 844 })
  await visit('/')
  await page.getByRole('button', { name: '打开导航' }).click()
  await page.waitForTimeout(250)
  await screenshot('mobile-navigation')
  await page.keyboard.press('Escape')
  await page.waitForTimeout(250)
  assert.equal(await page.locator('.sidebar').isVisible(), false)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.getByRole('button', { name: /今日精选/ }).click()
  assert.equal(await page.locator('.home-feed-disclosure').evaluate(el => getComputedStyle(el).transitionDuration), '0s')
  await page.emulateMedia({ reducedMotion: 'no-preference' })
  mode = 'long'
  for (const width of [390, 768, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 }); await visit('/feed')
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await screenshot(`${width}-long-feed`)
  }
  await page.setViewportSize({ width: 390, height: 844 })
  for (const state of ['empty', 'error', 'loading']) {
    mode = state
    for (const path of ['/feed', '/research', '/artifacts', '/memory', '/skills', '/approvals', '/mcp']) {
      await visit(path)
      await screenshot(`${state}-${path.slice(1)}`)
    }
  }
  mode = 'normal'
  await page.evaluate(() => localStorage.removeItem('authToken'))
  await page.addInitScript(() => localStorage.removeItem('authToken'))
  for (const width of [390, 768, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 }); await visit('/login'); await screenshot(`${width}-login`)
    await page.getByRole('tab', { name: '注册' }).click(); await page.waitForTimeout(250)
    assert.equal(await page.getByPlaceholder('可选').isVisible(), true)
    await screenshot(`${width}-register`)
  }
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ result: 'passed', responsivePages: 52, statePages: 21, screenshots: output, writes: writes.length }))
} finally { await browser.close() }
