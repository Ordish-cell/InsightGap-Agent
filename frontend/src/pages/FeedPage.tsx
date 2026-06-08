import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import * as feed from '../api/feed'
import type { FeedCard } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { ScoreBadge } from '../components/common/ScoreBadge'
import { StatusPill } from '../components/common/StatusPill'
import { relationLabel, sourceTypeLabel } from '../utils/labels'

const buckets = [
  { value: 'all', label: '全部' },
  { value: 'explicit_related', label: '显性相关' },
  { value: 'adjacent_domain', label: '邻近领域' },
  { value: 'far_domain', label: '远域启发' },
]

type FeedStats = { cards_count?: number; saved_count?: number; hidden_count?: number; average_final_score?: number }

const actionLabels: Record<string, string> = { save: '保存', useful: '有用', ignore: '忽略', not_relevant: '不相关' }

export function FeedPage() {
  const navigate = useNavigate()
  const [cards, setCards] = useState<FeedCard[]>([])
  const [bucket, setBucket] = useState('all')
  const [showAll, setShowAll] = useState(false)
  const [stats, setStats] = useState<FeedStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    setError('')
    try {
      const params: Record<string, unknown> = showAll ? { all: true } : {}
      if (bucket !== 'all') params.exposure_bucket = bucket
      const [nextCards, nextStats] = await Promise.all([feed.listCards(params), feed.stats()])
      setCards(nextCards)
      setStats(nextStats as FeedStats)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '信息雷达加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [bucket, showAll])

  async function research(cardId: number) {
    const result = await feed.startResearch(cardId) as { id?: string }
    if (result?.id) navigate(`/research/${result.id}`)
  }

  return (
    <section className="workbench-page feed-page">
      <PageHeader title="信息雷达" description="重点查看价值、证据、推荐原因和下一步行动。所有真实内容均来自后端 FeedCard。" actions={<><button className="button ghost" onClick={() => setShowAll((v) => !v)}>{showAll ? '只看今日' : '查看全部'}</button><button className="button" onClick={async () => { await feed.refresh(); await load() }}>刷新雷达</button></>} />
      <div className="metric-row">
        <div className="metric-card"><strong>{stats?.cards_count ?? cards.length}</strong><span>今日卡片</span></div>
        <div className="metric-card"><strong>{stats?.saved_count ?? 0}</strong><span>已保存</span></div>
        <div className="metric-card"><strong>{stats?.hidden_count ?? 0}</strong><span>已隐藏</span></div>
        <div className="metric-card"><strong>{Math.round((stats?.average_final_score ?? 0) * 100)}</strong><span>平均评分</span></div>
      </div>
      <div className="toolbar-panel">
        <div className="toolbar-left"><span className="toolbar-label">探索比例</span><StatusPill value="explicit_related" /><StatusPill value="adjacent_domain" /><StatusPill value="far_domain" /></div>
        <select value={bucket} onChange={(event) => setBucket(event.target.value)}>{buckets.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}</select>
      </div>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在整理信息差" /> : !cards.length ? <EmptyState title="暂无信息差卡片" description="点击刷新雷达，系统会重新抓取来源并生成信息差卡片。" /> : (
        <div className="feed-list">
          {cards.map((card) => (
            <article className="feed-item-card" key={card.id}>
              <div className="feed-item-top"><div className="row"><ScoreBadge score={card.final_score || 0} /><StatusPill value={card.exposure_bucket || card.relation_type} />{card.low_confidence ? <StatusPill value="低置信" /> : null}</div><span className="muted small">证据 {card.evidence?.length || 0} 条 · {sourceTypeLabel(card.source_type)} · {card.domain || '未标记域名'}</span></div>
              <div className="feed-item-body"><h2 title={card.original_title || card.title}>{card.title}</h2><p>{card.one_sentence_value || card.summary || '暂无摘要。'}</p><div className="info-two-col"><div className="soft-info-box"><strong>信息差</strong><span>{card.information_gap || '暂无信息差说明。'}</span></div><div className="soft-info-box muted-box"><strong>为什么与你有关</strong><span>{card.why_you || `当前归类为${relationLabel(card.exposure_bucket || card.relation_type)}，暂无更细画像原因。`}</span></div></div></div>
              <div className="feed-item-actions"><Link className="button secondary" to={`/feed/${card.id}`}>详情</Link><button className="button" onClick={() => research(card.id)}>深度研究</button>{Object.entries(actionLabels).map(([action, label]) => <button className="button ghost" key={action} onClick={() => feed.feedback(card.id, { action })}>{label}</button>)}</div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
