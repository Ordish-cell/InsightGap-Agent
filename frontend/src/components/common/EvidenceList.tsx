import type { EvidenceItem } from '../../api/types'
import { EmptyState } from './EmptyState'
import { ScoreBadge } from './ScoreBadge'

export function EvidenceList({ evidence }: { evidence?: EvidenceItem[] }) {
  if (!evidence?.length) return <EmptyState title="No evidence yet" description="Evidence will appear after research, RAG, or feed ingestion." />
  return (
    <div className="evidence-list">
      {evidence.map((item, index) => {
        const url = item.source_url || item.url
        return (
          <article className="evidence-item" key={`${item.title || item.chunk_id || index}`}>
            <div>
              <strong>{item.title || item.document_id || `Evidence ${index + 1}`}</strong>
              <p>{item.snippet || item.quote || item.summary || 'No snippet provided.'}</p>
              {url ? <a href={String(url)} target="_blank" rel="noreferrer">{String(url)}</a> : null}
            </div>
            {typeof item.score === 'number' ? <ScoreBadge score={item.score} /> : null}
          </article>
        )
      })}
    </div>
  )
}
