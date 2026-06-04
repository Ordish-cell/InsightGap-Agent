import { FormEvent, useEffect, useState } from 'react'

import * as memory from '../api/memory'
import { apiRequest } from '../api/client'
import type { MemoryItem, MemorySummary } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

type SummaryLike = MemorySummary & { total_count?: number; memory_count?: number; semantic_count?: number; episodic_count?: number }

interface GrowthCategory {
  category: string
  label: string
  count: number
  memories: Array<{
    id: number
    content: string
    importance: number
    effective_importance: number
    status: string
    stability: string
    confidence: number
    evidence_count: number
    last_seen_at: string
    superseded_by?: number
    supersedes?: number
  }>
}

interface GrowthProfile {
  categories: GrowthCategory[]
  semantic_count: number
  episodic_count: number
  recent_episodic: MemoryItem[]
}

export function MemoryPage() {
  const [summary, setSummary] = useState<SummaryLike | null>(null)
  const [items, setItems] = useState<MemoryItem[]>([])
  const [profile, setProfile] = useState<GrowthProfile | null>(null)
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState<'growth' | 'search'>('growth')

  async function load() {
    setLoading(true)
    try {
      const [s, p] = await Promise.all([
        memory.summary() as Promise<SummaryLike>,
        apiRequest<GrowthProfile>('/memory/growth-profile'),
      ])
      setSummary(s)
      setProfile(p)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '记忆加载失败')
    } finally {
      setLoading(false)
    }
  }

  async function loadSearch() {
    try {
      setItems(await memory.search({ query }))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '搜索失败')
    }
  }

  useEffect(() => { void load() }, [])
  async function submit(event: FormEvent) { event.preventDefault(); setView('search'); await loadSearch() }

  function statusLabel(s: string) {
    const map: Record<string, string> = { active: '生效中', superseded: '已替代', archived: '已归档', low_confidence: '低置信' }
    return map[s] || s
  }

  function statusClass(s: string) {
    const map: Record<string, string> = { active: 'active', superseded: 'superseded', archived: 'muted', low_confidence: 'muted' }
    return map[s] || ''
  }

  async function handleReflect() {
    await apiRequest('/memory/reflect', { method: 'POST', body: {} })
    await load()
  }

  async function handleForget(id: number) {
    await memory.forget({ memory_id: id })
    await load()
  }

  return (
    <section className="workbench-page memory-page">
      <PageHeader
        title="Agent 已记住的长期设定"
        description="Agent 从你的对话和行为中持续提炼长期设定。这里不是偏好设置页面——设定会随你的使用自动成长和演变。"
        actions={
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="button secondary" onClick={handleReflect}>整理碎片</button>
            <button className="button secondary" onClick={() => memory.consolidate().then(load)}>整理记忆</button>
          </div>
        }
      />
      <div className="metric-row">
        <div className="metric-card"><strong>{profile?.semantic_count ?? summary?.semantic_count ?? '-'}</strong><span>长期设定</span></div>
        <div className="metric-card"><strong>{profile?.episodic_count ?? summary?.episodic_count ?? '-'}</strong><span>行为事件</span></div>
        <div className="metric-card"><strong>{profile?.categories?.length ?? 0}</strong><span>设定分类</span></div>
      </div>
      <div className="view-tabs" style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button className={`button ${view === 'growth' ? '' : 'ghost'}`} onClick={() => setView('growth')}>长期设定</button>
        <button className={`button ${view === 'search' ? '' : 'ghost'}`} onClick={() => setView('search')}>搜索记忆</button>
      </div>
      {view === 'search' ? (
        <>
          <form className="search-panel" onSubmit={submit}>
            <input className="input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索记忆，比如：Agent、开源项目、信息差" />
            <button className="button">搜索</button>
          </form>
          {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载记忆" /> : !items.length ? (
            <EmptyState title="暂无匹配记忆" description="Agent 运行、深度研究和工具调用会逐步形成可控记忆。" />
          ) : (
            <div className="memory-list">
              {items.map((item) => (
                <article className="memory-card" key={item.id}>
                  <div className="memory-card-head">
                    <StatusPill value={item.memory_type} />
                    <span className="muted small">重要性 {item.importance ?? 0}</span>
                  </div>
                  <p>{item.content}</p>
                  <button className="button ghost" onClick={() => handleForget(item.id)}>忘记这条</button>
                </article>
              ))}
            </div>
          )}
        </>
      ) : (
        <>
          {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载长期设定" /> : !profile?.categories?.length ? (
            <EmptyState title="Agent 还没记住你的长期设定" description="多和 Agent 对话，让它帮你做研究、生成 Artifact、管理 Feed。设定会逐渐从你的行为中自动提炼。" />
          ) : (
            <div className="growth-categories">
              {profile.categories.map((cat) => (
                <div className="growth-category" key={cat.category}>
                  <div className="growth-category-head">
                    <h3>{cat.label}</h3>
                    <span className="muted small">{cat.count} 条</span>
                  </div>
                  <div className="growth-memory-list">
                    {cat.memories.map((mem) => (
                      <div className={`growth-memory-item ${statusClass(mem.status)}`} key={mem.id}>
                        <div className="growth-memory-top">
                          <span className={`mini-pill ${statusClass(mem.status)}`}>{statusLabel(mem.status)}</span>
                          <span className="muted small">
                            重要性 {Math.round(mem.effective_importance * 100)}%
                            {mem.evidence_count > 1 ? ` · 出现 ${mem.evidence_count} 次` : ''}
                          </span>
                        </div>
                        <p>{mem.content}</p>
                        <div className="growth-memory-actions">
                          <button className="button ghost small" onClick={() => handleForget(mem.id)}>忘记</button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  )
}
