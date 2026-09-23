import { FeedFeedback } from '../components/common/FeedFeedback'
import { ActionNotice } from '../components/common/ActionNotice'
import { useAction } from '../components/common/useAction'
import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import * as feed from '../api/feed'
import type { FeedCard } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { EvidenceList } from '../components/common/EvidenceList'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { ScoreBadge } from '../components/common/ScoreBadge'
import { StatusPill } from '../components/common/StatusPill'
import { sourceTypeLabel } from '../utils/labels'


export function FeedCardDetailPage() {
  const { cardId = '' } = useParams()
  const navigate = useNavigate()
  const action = useAction()
  const [card, setCard] = useState<FeedCard | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => { feed.getCard(cardId).then(setCard).catch((exc) => setError(exc.message)).finally(() => setLoading(false)) }, [cardId])

  async function research() {
    const result = await feed.startResearch(cardId) as { id?: string }
    if (!result?.id) throw new Error('未获取到研究任务，请重试。')
    navigate(`/research/${result.id}`)
  }

  if (loading) return <LoadingState title="正在加载信息卡片" />
  if (error) return <ErrorState message={error} />
  if (!card) return <EmptyState title="未找到信息卡片" />

  return (
    <section className="workbench-page">
      <PageHeader title={card.title} description={card.one_sentence_value || card.summary} actions={<><Link className="button secondary" to="/feed">返回信息流</Link><button className="button" disabled={action.busy} onClick={() => action.run(research)}>{action.busy ? '创建中…' : '深度研究'}</button></>} />
      <ActionNotice message={action.message} error={action.failed} />
      <div className="split">
        <div className="stack">
          <div className="panel insight-panel"><h2>信息差</h2><p>{card.information_gap || '暂无信息差说明。'}</p></div>
          <div className="panel why-panel"><h2>为什么与你有关</h2><p>{card.why_you || '暂无画像匹配说明。'}</p></div>
          <div className="panel"><h2>证据</h2><EvidenceList evidence={card.evidence} /></div>
          <div className="panel"><h2>建议行动</h2>{card.suggested_actions?.length ? <ul>{card.suggested_actions.map((item) => <li key={item}>{item}</li>)}</ul> : <EmptyState title="暂无建议行动" />}</div>
        </div>
        <aside className="stack">
          <div className="panel stack"><ScoreBadge score={card.final_score || 0} /><StatusPill value={card.exposure_bucket || card.relation_type} /><span className="muted small">{sourceTypeLabel(card.source_type)} · {card.domain || '未标记域名'}</span>{card.source_url ? <a className="button secondary" href={card.source_url} target="_blank" rel="noreferrer">打开来源</a> : null}</div>
          <details className="panel"><summary>评分详情</summary><JsonBlock value={card.score_detail} /></details>
          <div className="panel row"><FeedFeedback cardId={card.id} /></div>
        </aside>
      </div>
    </section>
  )
}
