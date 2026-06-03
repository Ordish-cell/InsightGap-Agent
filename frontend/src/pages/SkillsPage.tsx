import { useEffect, useState } from 'react'

import * as skills from '../api/skills'
import type { SkillDraft } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function SkillsPage() {
  const [items, setItems] = useState<SkillDraft[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  async function load() { setLoading(true); try { setItems(await skills.list()) } catch (exc) { setError(exc instanceof Error ? exc.message : 'Failed to load skills') } finally { setLoading(false) } }
  useEffect(() => { void load() }, [])
  return (
    <>
      <PageHeader title="Skills" description="Structured task recipes. They do not bypass approval gates or execute dangerous external writes." />
      {error ? <ErrorState message={error} /> : loading ? <LoadingState /> : !items.length ? <EmptyState title="No Skill drafts yet" /> : (
        <div className="grid">{items.map((skill) => (
          <article className="card stack" key={skill.id}>
            <div className="row"><StatusPill value={skill.status} /><StatusPill value={skill.safety_level} /></div>
            <h2>{skill.name}</h2><p>{skill.description}</p>
            <p className="muted small">{skill.trigger_text}</p>
            <JsonBlock value={{ input_schema: skill.input_schema, context_recipe: skill.context_recipe, tool_plan: skill.tool_plan, output_schema: skill.output_schema, eval_checks: skill.eval_checks }} />
            <div className="row"><button className="button secondary" onClick={() => skills.approve(skill.id).then(load)}>Approve</button><button className="button ghost" onClick={() => skills.disable(skill.id).then(load)}>Disable</button></div>
          </article>
        ))}</div>
      )}
    </>
  )
}
