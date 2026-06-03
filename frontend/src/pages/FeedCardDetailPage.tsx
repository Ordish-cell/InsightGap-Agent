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

export function FeedCardDetailPage() {
  const { cardId = '' } = useParams()
  const navigate = useNavigate()
  const [card, setCard] = useState<FeedCard | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    feed.getCard(cardId).then(setCard).catch((exc) => setError(exc.message)).finally(() => setLoading(false))
  }, [cardId])

  async function research() {
    const result = await feed.startResearch(cardId) as { id?: string }
    if (result?.id) navigate(`/research/${result.id}`)
  }

  if (loading) return <LoadingState title="Loading FeedCard" />
  if (error) return <ErrorState message={error} />
  if (!card) return <EmptyState title="FeedCard not found" />

  return (
    <>
      <PageHeader title={card.title} description={card.one_sentence_value || card.summary} actions={<><Link className="button secondary" to="/feed">Back to Feed</Link><button className="button" onClick={research}>Deep Research</button></>} />
      <div className="split">
        <div className="stack">
          <div className="panel insight-panel"><h2>Information gap</h2><p>{card.information_gap}</p></div>
          <div className="panel why-panel"><h2>Why you</h2><p>{card.why_you}</p></div>
          <div className="panel"><h2>Evidence</h2><EvidenceList evidence={card.evidence} /></div>
          <div className="panel"><h2>Suggested actions</h2>{card.suggested_actions?.length ? <ul>{card.suggested_actions.map((item) => <li key={item}>{item}</li>)}</ul> : <EmptyState title="No suggested actions" />}</div>
        </div>
        <aside className="stack">
          <div className="panel stack">
            <ScoreBadge score={card.final_score || 0} />
            <StatusPill value={card.exposure_bucket || card.relation_type} />
            <span className="muted small">{card.source_type} · {card.domain}</span>
            {card.source_url ? <a className="button secondary" href={card.source_url} target="_blank" rel="noreferrer">Open source</a> : null}
          </div>
          <div className="panel"><h2>Score breakdown</h2><JsonBlock value={card.score_detail} /></div>
          <div className="panel row">{['save', 'useful', 'ignore', 'not_relevant'].map((action) => <button className="button secondary" key={action} onClick={() => feed.feedback(card.id, { action })}>{action}</button>)}</div>
        </aside>
      </div>
    </>
  )
}
