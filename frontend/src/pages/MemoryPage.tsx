import { FormEvent, useEffect, useState } from 'react'

import * as memory from '../api/memory'
import type { MemoryItem, MemorySummary } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

type SummaryLike = MemorySummary & { total_count?: number; memory_count?: number; semantic_count?: number; episodic_count?: number }

export function MemoryPage() {
  const [summary, setSummary] = useState<SummaryLike | null>(null)
  const [items, setItems] = useState<MemoryItem[]>([])
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  async function load() {
    setLoading(true)
    try {
      setSummary(await memory.summary() as SummaryLike)
      setItems(await memory.search({ query }))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '记忆加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])
  async function submit(event: FormEvent) { event.preventDefault(); await load() }

  return (
    <section className="workbench-page memory-page">
      <PageHeader title="长期记忆" description="记忆必须可查看、可搜索、可删除。当前阶段不会伪造未开放的记忆开关。" actions={<button className="button secondary" onClick={() => memory.consolidate().then(load)}>整理记忆</button>} />
      <div className="metric-row"><div className="metric-card"><strong>{summary?.total_count ?? summary?.memory_count ?? items.length}</strong><span>记忆总数</span></div><div className="metric-card"><strong>{summary?.semantic_count ?? '-'}</strong><span>长期偏好</span></div><div className="metric-card"><strong>{summary?.episodic_count ?? '-'}</strong><span>事件记忆</span></div></div>
      <form className="search-panel" onSubmit={submit}><input className="input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索记忆，比如：Agent、开源项目、信息差" /><button className="button">搜索</button></form>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载记忆" /> : !items.length ? <EmptyState title="暂无匹配记忆" description="Agent 运行、深度研究和工具调用会逐步形成可控记忆。" /> : (
        <div className="memory-list">{items.map((item) => <article className="memory-card" key={item.id}><div className="memory-card-head"><StatusPill value={item.memory_type} /><span className="muted small">重要性 {item.importance ?? 0}</span></div><p>{item.content}</p><button className="button ghost" onClick={() => memory.forget({ memory_id: item.id }).then(load)}>忘记这条</button></article>)}</div>
      )}
    </section>
  )
}
