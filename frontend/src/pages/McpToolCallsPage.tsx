import { useEffect, useState } from 'react'

import * as mcp from '../api/mcp'
import type { McpTool, McpToolCall } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function McpToolCallsPage() {
  const [tools, setTools] = useState<McpTool[]>([])
  const [calls, setCalls] = useState<McpToolCall[]>([])
  const [selected, setSelected] = useState<McpTool | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    try {
      const nextTools = await mcp.listTools()
      const nextCalls = await mcp.listToolCalls().catch(() => [])
      setTools(nextTools)
      setCalls(nextCalls)
      setSelected(nextTools[0] || null)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '工具审计加载失败')
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { void load() }, [])

  return (
    <section className="workbench-page mcp-page">
      <PageHeader title="工具审计" description="展示工具注册、权限级别和关键调用记录。底层输入输出只放在折叠技术详情中。" actions={<button className="button secondary" onClick={load}>刷新</button>} />
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载工具审计" /> : (
        <div className="mcp-layout">
          <section className="mcp-tools-panel"><div className="section-title"><strong>可用工具</strong><span>{tools.length} 个</span></div><div className="tool-grid">{tools.map((tool) => <button className={selected?.name === tool.name ? 'tool-card active' : 'tool-card'} key={tool.name} onClick={() => setSelected(tool)}><strong>{tool.name}</strong><span>{tool.description}</span><StatusPill value={tool.safety_level} /></button>)}</div></section>
          <aside className="selected-tool-panel"><span className="eyebrow">Selected Tool</span><h2>{selected?.name || '未选择工具'}</h2><p>{selected?.description || '选择左侧工具查看说明。'}</p>{selected?.safety_level ? <StatusPill value={selected.safety_level} /> : null}<details><summary>技术详情</summary><JsonBlock value={selected} /></details></aside>
          <section className="tool-call-panel"><div className="section-title"><strong>调用记录</strong><span>{calls.length} 条</span></div>{!calls.length ? <EmptyState title="暂无工具调用" /> : <div className="call-list">{calls.map((call) => <article className="call-card" key={call.id}><div className="call-card-head"><StatusPill value={call.status} /><StatusPill value={call.safety_level} /><span className="muted small">user {call.user_id}</span></div><strong>{call.tool_name}</strong><p>{call.error ? `错误：${call.error}` : '调用已记录，输入输出已收起。'}</p><details><summary>技术详情</summary><JsonBlock value={{ input: call.input, output: call.output }} /></details></article>)}</div>}</section>
        </div>
      )}
    </section>
  )
}
