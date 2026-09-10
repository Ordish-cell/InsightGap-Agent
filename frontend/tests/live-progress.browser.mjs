// Isolated component browser acceptance. No real API or tools are called.
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const require = createRequire(import.meta.url)
const { chromium } = require('playwright')
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
const errors = []
page.on('pageerror', e => { errors.push(String(e)); console.error(String(e)) })
const origin = process.env.CHAT_UI_URL || 'http://127.0.0.1:5173'
await page.route('**/api/**', route => route.abort())
await page.route('**/__live_test', route => route.fulfill({ contentType: 'text/html', body: `
<html lang="zh"><meta charset="utf-8"><body><main style="max-width:800px;margin:60px auto;padding:24px"><p id="question">唉，我好累</p><div id="root"></div></main>
<script type="module">
import RefreshRuntime from '/@react-refresh';
RefreshRuntime.injectIntoGlobalHook(window); window.$RefreshReg$=()=>{};window.$RefreshSig$=()=>type=>type;window.__vite_plugin_react_preamble_installed__=true;
const React=(await import('/node_modules/.vite/deps/react.js')).default;
const {createRoot}=(await import('/node_modules/.vite/deps/react-dom_client.js')).default;
const {AgentThoughtStream}=await import('/src/components/agent/AgentThoughtStream.tsx');
await import('/src/styles/global.css');
const root=createRoot(document.getElementById('root'));
window.renderTrace=async (message)=>{const start=performance.now();root.render(React.createElement(React.Fragment,null,
React.createElement(AgentThoughtStream,{message,locale:'zh',onApprove:()=>{},onReject:()=>{}}),
React.createElement('p',{id:'answer'},message.content)));
await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));return performance.now()-start;};
</script></body></html>` }))
const e = (id, event_type, payload) => ({ id, event_seq: id, run_id: 8, event_type, payload })
const base = { message_id: 'isolated', role: 'assistant', run_id: 8, status: 'thinking', content: '', metadata: { interaction_version: 2 } }
const output = join(tmpdir(), 'insight-live-progress')
await mkdir(output, { recursive: true })
try {
  await page.goto(`${origin}/__live_test`)
  await page.waitForFunction(() => typeof window.renderTrace === 'function')
  await page.evaluate(m => window.renderTrace(m), base)
  await page.getByRole('status').filter({ hasText: '正在生成回复' }).waitFor()
  assert.equal(await page.locator('details').count(), 0)
  const chat = { ...base, status: 'streaming', content: '听起来你有些累了。现在想聊聊，还是先安静歇一会儿？', trace_events: [e(1, 'interaction_mode', { mode: 'chat' })] }
  await page.evaluate(m => window.renderTrace(m), chat)
  assert.equal(await page.getByText('风险等级').count(), 0)
  assert.equal(await page.locator('.activity-timeline').count(), 0)
  await page.screenshot({ path: join(output, 'chat.png') })
  const events = [e(1, 'interaction_mode', { mode: 'workflow' }), e(2, 'node_started', { step_id: 's1', display_name: '检查模型调用', status: 'running' })]
  const work = { ...base, trace_events: events }
  await page.evaluate(m => { document.querySelector('#question').textContent = '帮我分析项目为什么这么慢';return window.renderTrace(m) }, work)
  await page.getByRole('status').filter({ hasText: '检查模型调用' }).waitFor()
  assert.equal(await page.locator('details').getAttribute('open'), null)
  const times = []
  for (let i = 0; i < 30; i++) {
    const payload = { step_id: 's1', block_id: 'finding', text: `意图识别耗时 42 秒，回答生成尚未开始。样本 ${i + 1}` }
    events.push(e(i + 3, 'progress_completed', payload))
    times.push(await page.evaluate(m => window.renderTrace(m), work))
    assert.match(await page.locator('.live-progress-finding').textContent(), new RegExp(`样本 ${i + 1}`))
  }
  const p95 = times.sort((a, b) => a - b)[28]
  assert.ok(p95 <= 300, `event-to-paint p95 ${p95}ms`)
  await page.screenshot({ path: join(output, 'workflow.png') })
  await page.reload()
  await page.waitForFunction(() => typeof window.renderTrace === 'function')
  await page.evaluate(m => window.renderTrace(m), { ...work, trace_events: [...events, ...events] })
  assert.equal(await page.locator('.live-progress-finding').count(), 1)
  await page.setViewportSize({ width: 390, height: 844 })
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
  await page.screenshot({ path: join(output, 'workflow-mobile.png') })
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ passed: true, samples: 30, injected_event_to_paint_p95_ms: Math.round(p95), screenshots: output }))
} finally { await browser.close() }
