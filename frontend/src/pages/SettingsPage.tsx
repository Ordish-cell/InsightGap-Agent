import { FormEvent, useEffect, useState } from 'react'

import { me } from '../api/auth'
import type { CurrentUser } from '../api/types'
import { JsonBlock } from '../components/common/JsonBlock'
import { PageHeader } from '../components/common/PageHeader'

export function SettingsPage() {
  const [apiBaseUrl, setApiBaseUrl] = useState(localStorage.getItem('apiBaseUrl') || 'http://127.0.0.1:8000/api/v1')
  const [user, setUser] = useState<CurrentUser | null>(null)
  useEffect(() => { me().then(setUser).catch(() => setUser(null)) }, [])
  function save(event: FormEvent) { event.preventDefault(); localStorage.setItem('apiBaseUrl', apiBaseUrl) }
  return (
    <>
      <PageHeader title="Settings" description="Runtime configuration and governance rules for the MVP." />
      <div className="split">
        <div className="stack">
          <form className="panel stack" onSubmit={save}><label>API base URL<input className="input" value={apiBaseUrl} onChange={(event) => setApiBaseUrl(event.target.value)} /></label><button className="button" style={{ width: 'fit-content' }}>Save settings</button></form>
          <div className="panel"><h2>Feed scoring</h2><pre className="code-block">final_score ={"\n"}0.30 * personal_relevance{"\n"}+ 0.20 * novelty{"\n"}+ 0.15 * cross_domain_distance{"\n"}+ 0.15 * opportunity_value{"\n"}+ 0.10 * source_credibility{"\n"}+ 0.10 * actionability</pre></div>
          <div className="panel"><h2>Permission levels</h2><pre className="code-block">L0_READ_ONLY: read-only, may run automatically{"\n"}L1_DRAFT: create drafts only{"\n"}L2_LOCAL_WRITE: local writes such as Artifact, Memory, Skill{"\n"}L3_EXTERNAL_WRITE: approval required{"\n"}L4_HIGH_RISK: blocked by default</pre></div>
        </div>
        <aside className="panel"><h2>Current user</h2><JsonBlock value={user} /><p className="muted">This MVP does not perform real email sending, browser control, video comments, form submission, Exa, or Neo4j.</p></aside>
      </div>
    </>
  )
}
