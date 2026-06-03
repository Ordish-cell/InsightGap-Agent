import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import * as research from '../api/research'
import type { ResearchRun } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { EvidenceList } from '../components/common/EvidenceList'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function ResearchRunDetailPage() {
  const { researchRunId = '' } = useParams()
  const [run, setRun] = useState<ResearchRun | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    research.getRun(researchRunId).then(setRun).catch((exc) => setError(exc.message)).finally(() => setLoading(false))
  }, [researchRunId])

  if (loading) return <LoadingState title="Loading ResearchRun" />
  if (error) return <ErrorState message={error} />
  if (!run) return <EmptyState title="ResearchRun not found" />

  return (
    <>
      <PageHeader title="ResearchRun" description={run.query} actions={<StatusPill value={run.status} />} />
      <div className="split">
        <div className="stack">
          <div className="panel"><h2>Summary</h2><p>{run.summary || 'No summary yet.'}</p></div>
          <div className="panel"><h2>Report</h2>{run.markdown_report ? <pre className="markdown-block">{run.markdown_report}</pre> : <EmptyState title="No report yet" />}</div>
          <div className="panel"><h2>Evidence</h2><EvidenceList evidence={run.evidence} /></div>
        </div>
        <aside className="stack">
          <div className="panel stack">
            <span className="mono small">{run.id}</span>
            {run.artifact_id ? <Link className="button secondary" to={`/artifacts?artifactId=${run.artifact_id}`}>View artifact</Link> : <span className="muted">No linked Artifact</span>}
            {run.skill_draft_id ? <Link className="button secondary" to="/skills">View Skill draft</Link> : <span className="muted">No linked Skill draft</span>}
            {run.agent_run_id ? <Link className="button secondary" to="/agent">AgentRun #{run.agent_run_id}</Link> : null}
          </div>
          <div className="panel"><h2>Findings</h2><JsonBlock value={{ findings: run.findings, risks: run.risks, opportunities: run.opportunities, suggested_actions: run.suggested_actions }} /></div>
        </aside>
      </div>
    </>
  )
}
