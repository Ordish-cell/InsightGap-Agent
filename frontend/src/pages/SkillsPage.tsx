import { useEffect, useState } from 'react'

import * as skills from '../api/skills'
import type { SkillDraft } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function SkillsPage() {
  const [items, setItems] = useState<SkillDraft[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    try { setItems(await skills.list()) } catch (exc) { setError(exc instanceof Error ? exc.message : '技能加载失败') } finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [])

  return (
    <section className="workbench-page skills-page">
      <PageHeader title="技能库" description="技能是可复用任务配方，不会绕过审批，也不会自动执行危险外部操作。" />
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载技能" /> : !items.length ? <EmptyState title="暂无技能草稿" /> : (
        <div className="skill-grid">{items.map((skill) => <article className="skill-card" key={skill.id}><div className="skill-card-head"><StatusPill value={skill.status} /><StatusPill value={skill.safety_level} /></div><h2>{skill.name}</h2><p>{skill.description || '暂无描述。'}</p><div className="soft-info-box"><strong>触发方式</strong><span>{skill.trigger_text || '暂无触发描述'}</span></div><div className="skill-actions"><button className="button secondary" onClick={() => skills.approve(skill.id).then(load)}>批准</button><button className="button ghost" onClick={() => skills.disable(skill.id).then(load)}>停用</button></div></article>)}</div>
      )}
    </section>
  )
}
