import { useListMotion } from '../components/common/motion'
import { ActionNotice } from '../components/common/ActionNotice'
import { useAction } from '../components/common/useAction'
import { useEffect, useState } from 'react'

import * as skills from '../api/skills'
import type { SkillDraft } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function SkillsPage() {
  const action = useAction()
  const listMotion = useListMotion()
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
      <PageHeader title="技能库" description="将有效的任务方法保存下来，在需要时复用。" />
      <ActionNotice message={action.message} error={action.failed} />
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载技能" /> : !items.length ? <EmptyState title="暂无技能草稿" /> : (
        <div className="skill-grid" ref={listMotion}>{items.map((skill) => <article className="skill-card" key={skill.id} data-entry={skill.id}><div className="skill-card-head"><StatusPill value={skill.status} /><StatusPill value={skill.safety_level} /></div><h2>{skill.name}</h2><p>{skill.description || '暂无描述。'}</p><div className="soft-info-box"><strong>触发方式</strong><span>{skill.trigger_text || '暂无触发描述'}</span></div><div className="skill-actions">{skill.status !== 'approved' && <button className="button secondary" disabled={action.busy} onClick={() => action.run(async () => { await skills.approve(skill.id); await load() }, '技能已启用。')}>{skill.status === 'disabled' ? '启用' : '批准'}</button>}{skill.status !== 'disabled' && <button className="button ghost" disabled={action.busy} onClick={() => action.run(async () => { await skills.disable(skill.id); await load() }, '技能已停用。')}>停用</button>}</div></article>)}</div>
      )}
    </section>
  )
}
