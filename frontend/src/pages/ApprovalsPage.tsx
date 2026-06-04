import { useEffect, useState } from 'react'

import * as approvals from '../api/approvals'
import type { ApprovalItem } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function ApprovalsPage() {
  const [items, setItems] = useState<ApprovalItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    try { setItems(await approvals.list()) } catch (exc) { setError(exc instanceof Error ? exc.message : '审批事项加载失败') } finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [])

  return (
    <section className="workbench-page approvals-page">
      <PageHeader title="审批台" description="外部写入必须经过确认。高风险动作默认阻断，界面不会鼓励放行。" />
      <div className="policy-row"><div className="policy-card"><strong>L0</strong><span>只读自动</span></div><div className="policy-card"><strong>L1</strong><span>只生成草稿</span></div><div className="policy-card"><strong>L3</strong><span>必须审批</span></div><div className="policy-card"><strong>L4</strong><span>默认阻断</span></div></div>
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载审批事项" /> : !items.length ? <EmptyState title="暂无待审批事项" description="当 Agent 需要执行外部写入时，会在这里等待你确认。" /> : (
        <div className="approval-list">{items.map((item) => { const payload = item.payload || {}; const risk = String(payload.permission_level || payload.safety_level || item.approval_type || ''); const isL4 = risk.includes('L4'); return <article className="approval-card" key={item.id}><div className="approval-card-head"><div className="row"><StatusPill value={item.status} /><StatusPill value={risk || 'approval'} /></div><span className="muted small">run {item.run_id || 'none'}</span></div><h2>{item.title}</h2><p>{item.description}</p><div className="approval-actions"><button className="button secondary" disabled={isL4} onClick={() => approvals.approve(item.id).then(load)}>批准</button><button className="button danger" onClick={() => approvals.reject(item.id).then(load)}>拒绝</button></div></article> })}</div>
      )}
    </section>
  )
}
