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

  useEffect(() => { research.getRun(researchRunId).then(setRun).catch((exc) => setError(exc.message)).finally(() => setLoading(false)) }, [researchRunId])
  if (loading) return <LoadingState title="正在加载研究详情" />
  if (error) return <ErrorState message={error} />
  if (!run) return <EmptyState title="未找到研究记录" />

  return (
    <section className="workbench-page">
      <PageHeader title="研究详情" description={run.query} actions={<StatusPill value={run.status} />} />
      <div className="split">
        <div className="stack">
          <div className="panel"><h2>摘要</h2><p>{run.summary || '暂无摘要。'}</p></div>
          <div className="panel"><h2>研究报告</h2>{run.markdown_report ? <pre className="markdown-block">{run.markdown_report}</pre> : <EmptyState title="暂无报告" />}</div>
          <div className="panel"><h2>证据</h2><EvidenceList evidence={run.evidence} /></div>
        </div>
        <aside className="stack">
          <div className="panel stack"><span className="mono small">{run.id}</span>{run.artifact_id ? <Link className="button secondary" to={`/artifacts?artifactId=${run.artifact_id}`}>查看成果</Link> : <span className="muted">暂无关联成果</span>}{run.skill_draft_id ? <Link className="button secondary" to="/skills">查看技能草稿</Link> : <span className="muted">暂无技能草稿</span>}</div>
          <details className="panel"><summary>技术详情</summary><JsonBlock value={{ findings: run.findings, risks: run.risks, opportunities: run.opportunities, suggested_actions: run.suggested_actions }} /></details>
        </aside>
      </div>
    </section>
  )
}
