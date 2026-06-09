import { useEffect, useState } from 'react'

import type { AgentChatMessage, AgentEvent, UnknownRecord } from '../../api/types'
import { ApprovalCard, type ApprovalCardData } from './ApprovalCard'

type AgentThoughtStreamProps = {
  message: AgentChatMessage & { trace_events?: AgentEvent[] }
  locale: 'en' | 'zh'
  onApprove: (approvalId: number) => void
  onReject: (approvalId: number) => void
}

const zhText = {
  thinking: '正在思考',
  answering: '正在回答',
  completed: '已完成思考',
  failed: '思考中断',
  approve: '批准执行',
  reject: '拒绝',
}

function text(locale: 'en' | 'zh', zh: string, en: string) {
  return locale === 'zh' ? zh : en
}

function seconds(value?: number | null) {
  if (!value) return ''
  if (value < 1000) return `${Math.max(0.1, value / 1000).toFixed(1)}s`
  return `${(value / 1000).toFixed(value > 10000 ? 0 : 1)}s`
}

function asRecord(value: unknown): UnknownRecord {
  return value && typeof value === 'object' ? (value as UnknownRecord) : {}
}

function asThoughtText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  const item = asRecord(value)
  return String(item.text || item.summary || '').trim()
}

function collectVisibleThoughts(message: AgentThoughtStreamProps['message']) {
  const items: string[] = []
  const push = (value: unknown) => {
    const content = asThoughtText(value)
    if (content && !items.includes(content)) items.push(content)
  }

  const streamed = new Map<string, string>()
  ;(message.trace_events || []).forEach((event, index) => {
    const payload = asRecord(event.payload)
    if (event.event_type === 'visible_thought_delta') {
      const id = String(payload.id || `thought-${index}`)
      const current = streamed.get(id) || ''
      if (payload.status === 'completed' && payload.full_text) streamed.set(id, String(payload.full_text))
      else streamed.set(id, `${current}${String(payload.text || '')}`)
      return
    }
    if (event.event_type === 'visible_thought') push(payload.text || event.payload)
  })
  streamed.forEach((value) => push(value))

  const metadata = asRecord(message.metadata)
  const finalResponse = asRecord(metadata.final_response)
  const langgraphstatus = asRecord(message.langgraphstatus)
  const sources = [
    finalResponse.visible_thoughts,
    finalResponse.thinking_summary,
    metadata.visible_thoughts,
    langgraphstatus.visible_thoughts,
    (message as unknown as UnknownRecord).visible_thoughts,
  ]
  sources.forEach((source) => {
    if (Array.isArray(source)) source.forEach(push)
    else push(source)
  })

  return items
}

function approvalFrom(message: AgentThoughtStreamProps['message']): { approvalId: number; cardData: ApprovalCardData; currentStatus: string } {
  const events = message.trace_events || []

  // Find current approval state from event history (most recent wins)
  let approvalStatus = 'pending'
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i]
    if (!ev) continue
    const et = ev.event_type
    if (et === 'approval_granted' || et === 'approval_approved') { approvalStatus = 'approved'; break }
    if (et === 'approval_rejected') { approvalStatus = 'rejected'; break }
    if (et === 'tool_call_failed') { approvalStatus = 'approved'; break }
    if (et === 'tool_call_completed') { approvalStatus = 'approved'; break }
    if (et === 'run_completed') { approvalStatus = 'completed'; break }
  }
  if (message.status === 'completed') { approvalStatus = 'completed' }

  // Find approval_required event for payload data
  const event = events.find((item) => item.event_type === 'approval_required')
  let payload: UnknownRecord = {}
  if (event) {
    payload = asRecord(event.payload)
  } else {
    const meta = asRecord(message.metadata)
    const approvalPayload = asRecord(meta.approval_payload)
    if (approvalPayload.approval_id) { payload = approvalPayload }
  }
  if (!payload.approval_id && !payload.tool_name) return { approvalId: 0, cardData: {}, currentStatus: approvalStatus }

  const approvalId = Number(payload.approval_id || 0)
  const cardData: ApprovalCardData = {
    approval_id: payload.approval_id as string | number | undefined,
    run_id: payload.run_id as string | number | undefined,
    risk_level: String(payload.risk_level || 'L3'),
    tool_name: String(payload.tool_name || ''),
    title: String(payload.title || '需要你确认'),
    preview: asRecord(payload.preview),
    tool_args: asRecord(payload.tool_args),
    safety_notes: Array.isArray(payload.safety_notes) ? payload.safety_notes : [],
    actions: Array.isArray(payload.actions) ? payload.actions : ['approve', 'reject'],
    status: approvalStatus,
  }
  return { approvalId, cardData, currentStatus: approvalStatus }
}

export function AgentThoughtStream({ message, locale, onApprove, onReject }: AgentThoughtStreamProps) {
  const thoughts = collectVisibleThoughts(message)
  const status = String(message.status || 'completed')
  const running = ['thinking', 'running', 'created', 'queued'].includes(status)
  const waitingApproval = status === 'waiting_approval'
  const resuming = status === 'resuming'
  const answering = status === 'streaming'
  const failed = status === 'failed'
  const elapsed = seconds(message.elapsed_ms)
  const label = answering
    ? text(locale, zhText.answering, 'Answering')
    : waitingApproval
      ? '等待审批'
      : resuming
        ? '已批准，继续执行...'
        : running
          ? text(locale, zhText.thinking, 'Thinking')
          : failed
            ? text(locale, zhText.failed, 'Thinking interrupted')
            : text(locale, zhText.completed, 'Completed reasoning')
  const { approvalId, cardData, currentStatus } = approvalFrom(message)
  const [open, setOpen] = useState(running || failed || waitingApproval || resuming)

  useEffect(() => {
    setOpen(running || failed || waitingApproval || resuming)
  }, [failed, running, waitingApproval, resuming, message.message_id])

  if (!thoughts.length && !running && !failed && !answering && !approvalId) return null

  return (
    <div className={running ? 'agent-thought-stream running' : failed ? 'agent-thought-stream failed' : 'agent-thought-stream'}>
      <button className="thought-stream-status" type="button" onClick={() => setOpen((value) => !value)}>
        <span className={running ? 'thinking-dot active' : 'thinking-dot'} />
        <strong>
          {label}
          {elapsed ? ` · ${elapsed}` : ''}
        </strong>
        {thoughts.length || approvalId ? <span className="thought-stream-chevron">{open ? 'v' : '>'}</span> : null}
      </button>
      {open ? (
        <div className="thought-paragraphs">
          {thoughts.map((item, index) => <p key={`${index}-${item}`}>{item}</p>)}
        </div>
      ) : null}
      {approvalId && open ? (
        <ApprovalCard
          data={cardData}
          locale={locale}
          onApprove={onApprove}
          onReject={onReject}
        />
      ) : null}
    </div>
  )
}
