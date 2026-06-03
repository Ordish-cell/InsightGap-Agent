import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import * as feed from '../api/feed'
import type { FeedCard } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { ScoreBadge } from '../components/common/ScoreBadge'
import { StatusPill } from '../components/common/StatusPill'

const buckets = ['all', 'explicit_related', 'adjacent_domain', 'far_domain']

export function FeedPage() {
  const navigate = useNavigate()
  const [cards, setCards] = useState<FeedCard[]>([])
  const [bucket, setBucket] = useState('all')
  const [stats, setStats] = useState<unknown>(null)
  const [sources, setSources] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    setError('')
    try {
      const params = bucket === 'all' ? undefined : { exposure_bucket: bucket }
      const [nextCards, nextStats, nextSources] = await Promise.all([feed.listCards(params), feed.stats(), feed.sources()])
      setCards(nextCards)
      setStats(nextStats)
      setSources(nextSources)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Failed to load feed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [bucket])
  const grouped = useMemo(() => cards, [cards])

  async function research(cardId: number) {
    const result = await feed.startResearch(cardId) as { id?: string }
    if (result?.id) navigate(`/research/${result.id}`)
  }

  return (
    <>
      <PageHeader
        title="Feed"
        description="High-value information gaps, ranked by personal relevance, novelty, distance, opportunity value, source credibility, and actionability."
        actions={<button className="button" onClick={async () => { await feed.refresh(); await load() }}>Refresh Feed</button>}
      />
      <div className="panel row">
        <span>Feed mix:</span>
        <StatusPill value="30% explicit" /><StatusPill value="40% adjacent" /><StatusPill value="30% far" />
        <select value={bucket} onChange={(event) => setBucket(event.target.value)} style={{ maxWidth: 220 }}>
          {buckets.map((item) => <option key={item}>{item}</option>)}
        </select>
      </div>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="Loading feed" /> : !grouped.length ? <EmptyState title="No FeedCards yet" description="Refresh the feed to ingest sources and generate information-gap cards." /> : (
        <div className="grid">
          {grouped.map((card) => (
            <article className="card feed-card stack" key={card.id}>
              <div className="row">
                <ScoreBadge score={card.final_score || 0} />
                <StatusPill value={card.exposure_bucket || card.relation_type || 'unclassified'} />
                {card.low_confidence ? <StatusPill value="low_confidence" /> : null}
              </div>
              <h2>{card.title}</h2>
              <p>{card.one_sentence_value || card.summary || 'No summary provided.'}</p>
              <div className="insight-panel"><strong>Information gap</strong><p>{card.information_gap || 'No information gap provided.'}</p></div>
              <div className="why-panel"><strong>Why you</strong><p>{card.why_you || 'Profile match is not available yet.'}</p></div>
              <div className="small muted">Evidence {card.evidence?.length || 0} · {card.source_type || 'source'} · {card.domain || 'unknown domain'}</div>
              <div className="row">
                <Link className="button secondary" to={`/feed/${card.id}`}>Open detail</Link>
                <button className="button" onClick={() => research(card.id)}>Deep Research</button>
                {['save', 'useful', 'ignore', 'not_relevant'].map((action) => <button className="button ghost" key={action} onClick={() => feed.feedback(card.id, { action })}>{action}</button>)}
              </div>
            </article>
          ))}
        </div>
      )}
      <div className="split" style={{ marginTop: 16 }}>
        <div className="panel"><h2>Feed stats</h2><JsonBlock value={stats} /></div>
        <div className="panel"><h2>Sources</h2><JsonBlock value={sources} /></div>
      </div>
    </>
  )
}
