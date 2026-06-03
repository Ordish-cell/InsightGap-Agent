import { FormEvent, useEffect, useState } from 'react'

import * as memory from '../api/memory'
import type { MemoryItem, MemorySummary } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function MemoryPage() {
  const [summary, setSummary] = useState<MemorySummary | null>(null)
  const [items, setItems] = useState<MemoryItem[]>([])
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  async function load() {
    setLoading(true)
    try {
      setSummary(await memory.summary())
      setItems(await memory.search({ query }))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Failed to load memory')
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { void load() }, [])
  async function submit(event: FormEvent) { event.preventDefault(); await load() }

  return (
    <>
      <PageHeader title="Memory" description="Memory must stay inspectable and controllable. This stage supports viewing, searching, adding, consolidating, and backend-supported forget." actions={<button className="button secondary" onClick={() => memory.consolidate().then(load)}>Consolidate</button>} />
      <form className="panel row" onSubmit={submit}><input className="input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search memory" style={{ maxWidth: 420 }} /><button className="button">Search</button></form>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState /> : (
        <div className="split">
          <div className="stack">
            {!items.length ? <EmptyState title="No memory found" description="Add memory through Agent runs, Research, or MCP tools." /> : items.map((item) => (
              <article className="card stack" key={item.id}>
                <div className="row"><StatusPill value={item.memory_type} /><span className="muted small">importance {item.importance ?? 0}</span></div>
                <p>{item.content}</p>
                <button className="button ghost" onClick={() => memory.forget({ memory_id: item.id }).then(load)}>Forget</button>
              </article>
            ))}
          </div>
          <aside className="panel"><h2>Summary</h2><JsonBlock value={summary} /><p className="muted small">关闭某类记忆的策略开关尚未开放，当前阶段不伪造该控制。</p></aside>
        </div>
      )}
    </>
  )
}
