import { useListMotion } from '../components/common/motion'
import { ActionNotice } from '../components/common/ActionNotice'
import { useAction } from '../components/common/useAction'
import { useEffect, useState } from 'react'

import * as approvals from '../api/approvals'
import type { ApprovalItem } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function ApprovalsPage() {
  const action = useAction()
  const listMotion = useListMotion()
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
      <PageHeader title="审批台" description="查看待处理的操作，确认执行范围与结果。" />
      <details className="policy-details"><summary>操作权限说明</summary><div className="policy-row"><div className="policy-card"><strong>L0</strong><span>只读自动</span></div><div className="policy-card"><strong>L1</strong><span>只生成草稿</span></div><div className="policy-card"><strong>L3</strong><span>必须审批</span></div><div className="policy-card"><strong>L4</strong><span>默认阻断</span></div></div></details>
      <ActionNotice message={action.message} error={action.failed} />
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载审批事项" /> : !items.length ? <EmptyState title="暂无待审批事项" description="当 Agent 需要执行外部写入时，会在这里等待你确认。" /> : (
        <div className="approval-list" ref={listMotion}>{items.map((item) => { const payload = item.payload || {}; const risk = String(payload.permission_level || payload.safety_level || item.approval_type || ''); const isL4 = risk.includes('L4'); return <article className="approval-card" key={item.id} data-entry={item.id}><div className="approval-card-head"><div className="row"><StatusPill value={item.status} /><StatusPill value={risk || 'approval'} /></div></div><h2>{item.title}</h2><p>{item.description}</p>{item.status === 'pending' && <div className="approval-actions"><button className="button secondary" disabled={isL4 || action.busy} title={isL4 ? 'L4 高风险操作默认阻断' : undefined} onClick={() => action.run(async () => { await approvals.approve(item.id); await load() }, '批准已提交，请查看最新执行状态。')}>批准</button><button className="button danger" disabled={action.busy} onClick={() => action.run(async () => { await approvals.reject(item.id); await load() }, '已拒绝该操作。')}>拒绝</button>{isL4 && <span className="muted small">L4 高风险操作默认阻断</span>}</div>}<details><summary>执行详情</summary><pre className="json-block">{JSON.stringify(item, null, 2)}</pre></details></article> })}</div>
      )}
    </section>
  )
}
