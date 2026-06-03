import { FormEvent, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import * as research from '../api/research'
import type { ResearchRun } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function ResearchRunsPage() {
  const navigate = useNavigate()
  const [runs, setRuns] = useState<ResearchRun[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    try { setRuns(await research.listRuns()) } catch (exc) { setError(exc instanceof Error ? exc.message : 'Failed to load research runs') } finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [])

  async function submit(event: FormEvent) {
    event.preventDefault()
    const run = await research.createRun({ query, depth: 'standard' })
    navigate(`/research/${run.id}`)
  }

  return (
    <>
      <PageHeader title="Research" description="Create and inspect Deep Research runs generated from FeedCards or independent queries." />
      <form className="panel stack" onSubmit={submit}>
        <label>Research query<textarea className="textarea" value={query} onChange={(event) => setQuery(event.target.value)} required /></label>
        <button className="button" style={{ width: 'fit-content' }}>Create ResearchRun</button>
      </form>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState /> : !runs.length ? <EmptyState title="No ResearchRuns yet" /> : (
        <div className="stack">
          {runs.map((run) => (
            <Link className="card stack" to={`/research/${run.id}`} key={run.id}>
              <div className="row"><StatusPill value={run.status} /><span className="mono small">{run.id}</span></div>
              <strong>{run.query || 'Untitled research'}</strong>
              <span className="muted small">Artifact {run.artifact_id || 'none'} · Completed {run.completed_at || 'pending'}</span>
            </Link>
          ))}
        </div>
      )}
    </>
  )
}
