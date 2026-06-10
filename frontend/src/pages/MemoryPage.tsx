import { FormEvent, useCallback, useEffect, useState } from 'react'

import { apiRequest } from '../api/client'
import * as memory from '../api/memory'
import type { LongTermMemoryItem, LongTermMemoryListResponse, MemoryItem, MemorySummary } from '../api/types'
import { ConfirmModal } from '../components/common/ConfirmModal'
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

type TabId = 'long-term' | 'growth' | 'search'

export function MemoryPage() {
  const [tab, setTab] = useState<TabId>('long-term')
  const [summary, setSummary] = useState<SummaryLike | null>(null)
  const [profile, setProfile] = useState<GrowthProfile | null>(null)
  const [query, setQuery] = useState('')
  const [searchItems, setSearchItems] = useState<MemoryItem[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  // ── Long-term tab state ──
  const [ltItems, setLtItems] = useState<LongTermMemoryItem[]>([])
  const [ltTotal, setLtTotal] = useState(0)
  const [ltPage, setLtPage] = useState(1)
  const [ltPageSize] = useState(20)
  const [ltType, setLtType] = useState('')
  const [ltCategory, setLtCategory] = useState('')
  const [ltStatus, setLtStatus] = useState('')
  const [ltQuery, setLtQuery] = useState('')
  const [ltLoading, setLtLoading] = useState(false)
  const [ltError, setLtError] = useState('')

  // ── Confirm modal ──
  const [confirm, setConfirm] = useState<{ title: string; message: string; danger?: boolean; action: () => void } | null>(null)

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

  const loadLongTerm = useCallback(async (page?: number, search?: boolean) => {
    setLtLoading(true)
    setLtError('')
    const p = page ?? ltPage
    try {
      const params: Record<string, unknown> = { page: p, page_size: ltPageSize }
      if (ltType) params.type = ltType
      if (ltCategory) params.category = ltCategory
      if (ltStatus) params.status = ltStatus
      if (ltQuery) params.query = ltQuery

      const result = search
        ? await memory.searchLongTermMemories(params)
        : await memory.listLongTermMemories(params)
      setLtItems(result.items || [])
      setLtTotal(result.total || 0)
      setLtPage(result.page || p)
    } catch (exc) {
      setLtError(exc instanceof Error ? exc.message : '长期记忆加载失败')
    } finally {
      setLtLoading(false)
    }
  }, [ltPage, ltPageSize, ltType, ltCategory, ltStatus, ltQuery])

  async function loadSearch() {
    try {
      setSearchItems(await memory.search({ query }))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '搜索失败')
    }
  }

  useEffect(() => { void load() }, [])
  useEffect(() => {
    if (tab === 'long-term') void loadLongTerm(1, !!ltQuery)
  }, [tab, ltType, ltCategory, ltStatus]) // eslint-disable-line react-hooks/exhaustive-deps

  async function submit(event: FormEvent) { event.preventDefault(); setTab('search'); await loadSearch() }
  function handleLtSearch(e: FormEvent) { e.preventDefault(); void loadLongTerm(1, true) }

  function statusLabel(s: string) {
    const map: Record<string, string> = { active: '生效中', superseded: '已替代', archived: '已归档', low_confidence: '低置信' }
    return map[s] || s
  }

  function statusClass(s: string) {
    const map: Record<string, string> = { active: 'active', superseded: 'superseded', archived: 'muted', low_confidence: 'muted' }
    return map[s] || ''
  }

  async function handleReflect() { await apiRequest('/memory/reflect', { method: 'POST', body: {} }); await load() }
  async function handleForget(id: number) { await memory.forget({ memory_id: id }); await load() }

  // ── Long-term actions ──
  async function handleArchive(id: number) {
    try { await memory.archiveMemory(id); void loadLongTerm() } catch (e) { setLtError(e instanceof Error ? e.message : '归档失败') }
  }
  async function handleRestore(id: number) {
    try { await memory.restoreMemory(id); void loadLongTerm() } catch (e) { setLtError(e instanceof Error ? e.message : '恢复失败') }
  }
  function handleDeleteClick(id: number) {
    setConfirm({
      title: '删除记忆', message: '确定要永久删除这条记忆吗？此操作不可撤销。', danger: true,
      action: async () => { try { await memory.deleteMemory(id); void loadLongTerm() } catch (e) { setLtError(e instanceof Error ? e.message : '删除失败') } finally { setConfirm(null) } },
    })
  }
  async function handleImportanceChange(id: number, value: number) {
    try { await memory.updateMemory(id, { importance: Math.max(0, Math.min(1, value)) }); void loadLongTerm() } catch (e) { /* ignore */ }
  }

  const totalPages = Math.max(1, Math.ceil(ltTotal / ltPageSize))

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
        <button className={`button ${tab === 'long-term' ? '' : 'ghost'}`} onClick={() => setTab('long-term')}>长期记忆</button>
        <button className={`button ${tab === 'growth' ? '' : 'ghost'}`} onClick={() => setTab('growth')}>长期设定</button>
        <button className={`button ${tab === 'search' ? '' : 'ghost'}`} onClick={() => setTab('search')}>搜索记忆</button>
      </div>

      {/* ── Long-Term Tab ── */}
      {tab === 'long-term' && (
        <>
          {/* Filter bar */}
          <div className="lt-filter-bar" style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <select className="input" value={ltType} onChange={e => { setLtType(e.target.value); setLtPage(1) }} style={{ width: 120 }}>
              <option value="">全部类型</option>
              <option value="semantic">长期设定</option>
              <option value="episodic">行为事件</option>
            </select>
            <select className="input" value={ltCategory} onChange={e => { setLtCategory(e.target.value); setLtPage(1) }} style={{ width: 140 }}>
              <option value="">全部分类</option>
              <option value="tech_stack">技术栈</option>
              <option value="preference">产品偏好</option>
              <option value="project_goal">项目目标</option>
              <option value="boundary">当前边界</option>
              <option value="workflow_pattern">任务模式</option>
              <option value="product_principle">产品原则</option>
              <option value="uncategorized">其他</option>
            </select>
            <select className="input" value={ltStatus} onChange={e => { setLtStatus(e.target.value); setLtPage(1) }} style={{ width: 120 }}>
              <option value="">全部状态</option>
              <option value="active">生效中</option>
              <option value="archived">已归档</option>
              <option value="superseded">已替代</option>
              <option value="low_confidence">低置信</option>
            </select>
            <form onSubmit={handleLtSearch} style={{ display: 'flex', gap: 8, flex: 1, minWidth: 200 }}>
              <input className="input" value={ltQuery} onChange={e => setLtQuery(e.target.value)} placeholder="搜索记忆内容..." style={{ flex: 1 }} />
              <button className="button secondary" type="submit">搜索</button>
              {ltQuery && <button className="button ghost" type="button" onClick={() => { setLtQuery(''); void loadLongTerm(1, false) }}>清除</button>}
            </form>
          </div>

          {ltError && <ErrorState message={ltError} />}
          {ltLoading ? <LoadingState title="正在加载长期记忆" /> : !ltItems.length ? (
            <EmptyState title="暂无长期记忆" description="Agent 运行、深度研究和工具调用会逐步形成可控记忆。" />
          ) : (
            <>
              <div className="memory-list">
                {ltItems.map((item) => (
                  <article className="memory-card" key={item.id}>
                    <div className="memory-card-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                        <StatusPill value={item.memory_type} />
                        {item.category && <StatusPill value={item.category} />}
                        <StatusPill value={item.status || 'active'} />
                      </div>
                      <div className="muted small" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                        {item.confidence !== undefined && <span>置信度 {Math.round((item.confidence || 0) * 100)}%</span>}
                        {item.evidence_count !== undefined && item.evidence_count > 1 && <span>出现 {item.evidence_count} 次</span>}
                        {item.last_seen_at && <span>最近 {item.last_seen_at.slice(0, 10)}</span>}
                      </div>
                    </div>
                    <p style={{ margin: '8px 0' }}>{item.content}</p>
                    <div className="memory-card-foot" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <span className="muted small">重要性</span>
                        <input
                          type="number" min={0} max={1} step={0.05}
                          className="input" style={{ width: 72, textAlign: 'center' }}
                          value={item.importance ?? 0}
                          onChange={e => handleImportanceChange(item.id, parseFloat(e.target.value) || 0)}
                        />
                        <span className="muted small">有效 {(item.effective_importance ?? 0).toFixed(2)}</span>
                        {item.stability && <span className="muted small">稳定性 {item.stability}</span>}
                      </div>
                      <div style={{ display: 'flex', gap: 6 }}>
                        {item.status === 'archived' || item.status === 'superseded' ? (
                          <button className="button ghost small" onClick={() => handleRestore(item.id)}>恢复</button>
                        ) : (
                          <button className="button ghost small" onClick={() => handleArchive(item.id)}>归档</button>
                        )}
                        <button className="button ghost small danger" onClick={() => handleDeleteClick(item.id)}>删除</button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
              {ltTotal > ltPageSize && (
                <div className="pagination" style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 16, alignItems: 'center' }}>
                  <button className="button ghost" disabled={ltPage <= 1} onClick={() => loadLongTerm(ltPage - 1, !!ltQuery)}>上一页</button>
                  <span className="muted small">{ltPage} / {totalPages}（共 {ltTotal} 条）</span>
                  <button className="button ghost" disabled={ltPage >= totalPages} onClick={() => loadLongTerm(ltPage + 1, !!ltQuery)}>下一页</button>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ── Search Tab ── */}
      {tab === 'search' && (
        <>
          <form className="search-panel" onSubmit={submit}>
            <input className="input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索记忆，比如：Agent、开源项目、信息差" />
            <button className="button">搜索</button>
          </form>
          {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载记忆" /> : !searchItems.length ? (
            <EmptyState title="暂无匹配记忆" description="Agent 运行、深度研究和工具调用会逐步形成可控记忆。" />
          ) : (
            <div className="memory-list">
              {searchItems.map((item) => (
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
      )}

      {/* ── Growth Tab ── */}
      {tab === 'growth' && (
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

      {/* ── Confirm Modal ── */}
      <ConfirmModal
        open={!!confirm}
        title={confirm?.title || ''}
        message={confirm?.message || ''}
        danger={confirm?.danger}
        onConfirm={() => confirm?.action()}
        onCancel={() => setConfirm(null)}
      />
    </section>
  )
}
