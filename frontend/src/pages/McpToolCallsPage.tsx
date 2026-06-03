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
  const [health, setHealth] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  async function load() { setLoading(true); try { const [t, c, h] = await Promise.all([mcp.listTools(), mcp.listToolCalls(), mcp.health()]); setTools(t); setCalls(c); setHealth(h); setSelected(t[0] || null) } catch (exc) { setError(exc instanceof Error ? exc.message : 'Failed to load MCP') } finally { setLoading(false) } }
  useEffect(() => { void load() }, [])
  return (
    <>
      <PageHeader title="MCP Tools" description="Governance surface for local deterministic tools, permission levels, requests, results, and blocked states." actions={<button className="button secondary" onClick={load}>Refresh</button>} />
      {error ? <ErrorState message={error} /> : loading ? <LoadingState /> : (
        <div className="split">
          <div className="stack">
            <div className="panel"><h2>Health</h2><JsonBlock value={health} /></div>
            <div className="panel stack"><h2>Tools</h2>{tools.map((tool) => <button className="card" key={tool.name} onClick={() => setSelected(tool)}><strong>{tool.name}</strong><p><StatusPill value={tool.safety_level} /> {tool.description}</p></button>)}</div>
            <div className="panel stack"><h2>Tool calls</h2>{!calls.length ? <EmptyState title="No ToolCalls yet" /> : calls.map((call) => <article className="card stack" key={call.id}><div className="row"><StatusPill value={call.status} /><StatusPill value={call.safety_level} /><span className="mono small">user {call.user_id}</span></div><strong>{call.tool_name}</strong><JsonBlock value={{ input: call.input, output: call.output, error: call.error }} /></article>)}</div>
          </div>
          <aside className="panel"><h2>Tool schema</h2>{selected ? <JsonBlock value={selected} /> : <EmptyState title="No tool selected" />}</aside>
        </div>
      )}
    </>
  )
}
