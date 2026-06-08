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
    try { setRuns(await research.listRuns()) } catch (exc) { setError(exc instanceof Error ? exc.message : '研究任务加载失败') } finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [])

  async function submit(event: FormEvent) {
    event.preventDefault()
    const run = await research.createRun({ query, depth: 'standard', source: 'manual' })
    navigate(`/research/${run.id}`)
  }

  return (
    <section className="workbench-page research-page">
      <PageHeader title="深度研究" description="把一个信息差、问题或项目变成有证据、有风险判断、有行动建议的研究报告。" />
      <form className="hero-input-panel" onSubmit={submit}>
        <label><span>研究问题</span><textarea className="textarea large" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：某个开源 Agent 项目是否值得二开？机会和风险是什么？" required /></label>
        <div className="hero-input-footer"><span className="muted small">建议输入具体对象、判断目标和期望产出。</span><button className="button">创建研究</button></div>
      </form>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载研究记录" /> : !runs.length ? <EmptyState title="暂无研究记录" description="你可以从信息雷达发起研究，也可以在这里直接创建。" /> : (
        <div className="record-list">{runs.map((run) => <Link className="record-card" to={`/research/${run.id}`} key={run.id}><div className="record-main"><div className="row"><StatusPill value={run.status} /><span className="muted small">{run.completed_at || '等待完成'}</span></div><h2>{run.query || '未命名研究'}</h2><p>成果 {run.artifact_id ? '已生成' : '未生成'} · 技能草稿 {run.skill_draft_id ? '已创建' : '暂无'}</p></div><span className="record-arrow">→</span></Link>)}</div>
      )}
    </section>
  )
}
