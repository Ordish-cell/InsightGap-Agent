import { useEffect, useState } from 'react'

import * as approvals from '../api/approvals'
import type { ApprovalItem } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function ApprovalsPage() {
  const [items, setItems] = useState<ApprovalItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  async function load() { setLoading(true); try { setItems(await approvals.list()) } catch (exc) { setError(exc instanceof Error ? exc.message : 'Failed to load approvals') } finally { setLoading(false) } }
  useEffect(() => { void load() }, [])
  return (
    <>
      <PageHeader title="Approvals" description="L3 external writes require approval. L4 high-risk actions are blocked by default and should not be approved in the UI." />
      <div className="panel row"><StatusPill value="L3 approval" /><StatusPill value="L4 blocked" /></div>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState /> : !items.length ? <EmptyState title="No pending approvals" /> : (
        <div className="stack">{items.map((item) => {
          const payload = item.payload || {}
          const risk = String(payload.permission_level || payload.safety_level || item.approval_type || '')
          const isL4 = risk.includes('L4')
          return (
            <article className="card stack" key={item.id}>
              <div className="row"><StatusPill value={item.status} /><StatusPill value={risk || 'approval'} /><span className="mono small">run {item.run_id || 'none'}</span></div>
              <h2>{item.title}</h2><p>{item.description}</p><JsonBlock value={payload} />
              <div className="row"><button className="button secondary" disabled={isL4} onClick={() => approvals.approve(item.id).then(load)}>Approve</button><button className="button danger" onClick={() => approvals.reject(item.id).then(load)}>Reject</button></div>
            </article>
          )
        })}</div>
      )}
    </>
  )
}
