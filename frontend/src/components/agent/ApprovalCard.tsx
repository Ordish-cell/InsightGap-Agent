import { useEffect, useState } from 'react'
import type { UnknownRecord } from '../../api/types'

export type ApprovalCardData = {
  approval_id?: number | string
  run_id?: number | string
  risk_level?: string
  tool_name?: string
  title?: string
  preview?: UnknownRecord
  tool_args?: UnknownRecord
  safety_notes?: string[]
  actions?: string[]
  status?: string  // pending | approved | rejected | completed
}

type ApprovalCardProps = {
  data: ApprovalCardData
  onApprove: (approvalId: number) => void
  onReject: (approvalId: number) => void
  locale?: 'en' | 'zh'
}

const zhMap = {
  confirmTitle: '需要你确认',
  riskLevel: '风险等级',
  tool: '工具',
  preview: '操作预览',
  to: '收件人',
  subject: '主题',
  body: '正文',
  path: '路径',
  contentPreview: '内容预览',
  chars: '字符数',
  safetyNote: '安全提示',
  approve: '同意执行',
  reject: '拒绝',
  approving: '执行中...',
  approved: '已同意',
  rejected: '已拒绝',
  l4Blocked: '此操作风险太高，已被系统阻止。',
  sending: '发送中...',
  writing: '写入中...',
}

function t(locale: 'en' | 'zh', zh: string, en: string) {
  return locale === 'zh' ? zh : en
}

function asRecord(value: unknown): UnknownRecord {
  return value && typeof value === 'object' ? (value as UnknownRecord) : {}
}

function riskBadge(level: string) {
  const colors: Record<string, string> = {
    L0: '#6b7280', L1: '#10b981', L2: '#f59e0b', L3: '#f97316', L4: '#ef4444',
    L0_READ_ONLY: '#6b7280', L1_DRAFT: '#10b981', L2_LOCAL_WRITE: '#f59e0b',
    L3_EXTERNAL_WRITE: '#f97316', L4_HIGH_RISK: '#ef4444',
  }
  return colors[level] || '#6b7280'
}

export function ApprovalCard({ data, onApprove, onReject, locale = 'zh' }: ApprovalCardProps) {
  const approvalId = Number(data.approval_id || 0)
  const riskLevel = String(data.risk_level || 'L3')
  const isL4 = riskLevel.includes('L4')
  const [status, setStatus] = useState<'pending' | 'approving' | 'approved' | 'rejected' | 'completed'>(
    data.status === 'approved' || data.status === 'completed' ? 'approved' : data.status === 'rejected' ? 'rejected' : 'pending'
  )

  // Sync status when data.status changes (e.g. after resume events arrive)
  useEffect(() => {
    if (data.status === 'approved' || data.status === 'completed') setStatus('approved')
    else if (data.status === 'rejected') setStatus('rejected')
  }, [data.status])

  const preview = asRecord(data.preview)
  const toolName = String(data.tool_name || '')

  async function handleApprove() {
    if (!approvalId) return
    setStatus('approving')
    try {
      await onApprove(approvalId)
      setStatus('approved')
    } catch {
      setStatus('pending')
    }
  }

  async function handleReject() {
    if (!approvalId) return
    setStatus('rejected')
    try {
      await onReject(approvalId)
    } catch {
      setStatus('pending')
    }
  }

  function renderPreview() {
    if (!preview || Object.keys(preview).length === 0) return null

    // Email preview
    if (toolName.includes('email') || preview.to || preview.subject) {
      return (
        <div className="approval-preview">
          <h4>{t(locale, zhMap.preview, 'Preview')}</h4>
          {preview.to !== undefined ? (
            <div className="approval-preview-row">
              <span className="label">{t(locale, zhMap.to, 'To')}</span>
              <span>{String(preview.to)}</span>
            </div>
          ) : null}
          {preview.subject !== undefined ? (
            <div className="approval-preview-row">
              <span className="label">{t(locale, zhMap.subject, 'Subject')}</span>
              <span>{String(preview.subject)}</span>
            </div>
          ) : null}
          {preview.body !== undefined ? (
            <div className="approval-preview-row body">
              <span className="label">{t(locale, zhMap.body, 'Body')}</span>
              <pre>{String(preview.body).slice(0, 500)}</pre>
            </div>
          ) : null}
        </div>
      )
    }

    // File operation preview
    if (toolName.includes('local_file') || preview.path) {
      return (
        <div className="approval-preview">
          <h4>{t(locale, zhMap.preview, 'Preview')}</h4>
          {preview.path !== undefined ? (
            <div className="approval-preview-row">
              <span className="label">{t(locale, zhMap.path, 'Path')}</span>
              <code>{String(preview.path)}</code>
            </div>
          ) : null}
          {preview.content_preview !== undefined ? (
            <div className="approval-preview-row body">
              <span className="label">{t(locale, zhMap.contentPreview, 'Content')}</span>
              <pre>{String(preview.content_preview).slice(0, 300)}</pre>
              {preview.chars !== undefined ? (
                <small>{t(locale, zhMap.chars, 'Chars')}: {String(preview.chars)}</small>
              ) : null}
            </div>
          ) : null}
          {preview.warning !== undefined ? (
            <div className="approval-danger-note">{String(preview.warning)}</div>
          ) : null}
        </div>
      )
    }

    // Generic preview
    return (
      <div className="approval-preview">
        <h4>{t(locale, zhMap.preview, 'Preview')}</h4>
        <pre className="approval-raw-preview">{JSON.stringify(preview, null, 2)}</pre>
      </div>
    )
  }

  // L4 blocked — always show blocked state
  if (isL4) {
    return (
      <div className="approval-card danger blocked">
        <div className="approval-card-header danger">
          <span className="risk-badge" style={{ background: riskBadge(riskLevel) }}>{riskLevel}</span>
          <strong>{t(locale, zhMap.l4Blocked, 'This action is too risky and has been blocked by the system.')}</strong>
        </div>
        <div className="approval-detail">
          <p>{String(data.title || `${toolName}: 高危操作，已自动阻止`)}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`approval-card ${status}`}>
      <div className="approval-card-header">
        <span className="risk-badge" style={{ background: riskBadge(riskLevel) }}>{riskLevel}</span>
        <strong>{String(data.title || t(locale, zhMap.confirmTitle, 'Confirmation Required'))}</strong>
        {toolName ? (
          <code className="tool-name-tag">{toolName}</code>
        ) : null}
      </div>

      <div className="approval-detail">
        {renderPreview()}

        {data.safety_notes && data.safety_notes.length > 0 ? (
          <div className="approval-safety-notes">
            <h4>{t(locale, zhMap.safetyNote, 'Safety Notes')}</h4>
            <ul>
              {data.safety_notes.map((note, i) => (
                <li key={i}>{note}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      <div className="approval-actions">
        {status === 'pending' ? (
          <>
            <button className="approval-btn approve" type="button" onClick={handleApprove}>
              {t(locale, zhMap.approve, 'Approve')}
            </button>
            <button className="approval-btn reject" type="button" onClick={handleReject}>
              {t(locale, zhMap.reject, 'Reject')}
            </button>
          </>
        ) : status === 'approving' ? (
          <span className="approval-status">{t(locale, zhMap.approving, 'Executing...')}</span>
        ) : status === 'approved' || status === 'completed' ? (
          <span className="approval-status approved">{t(locale, zhMap.approved, 'Approved')}</span>
        ) : (
          <span className="approval-status rejected">{t(locale, zhMap.rejected, 'Rejected')}</span>
        )}
      </div>
    </div>
  )
}
