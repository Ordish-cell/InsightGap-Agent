import { useEffect, useMemo, useState } from 'react'

import type { AgentChatMessage, AgentEvent, UnknownRecord } from '../../api/types'
import { ApprovalCard, type ApprovalCardData } from './ApprovalCard'

type AgentThoughtStreamProps = {
  message: AgentChatMessage & { trace_events?: AgentEvent[] }
  locale: 'en' | 'zh'
  onApprove: (approvalId: number) => void
  onReject: (approvalId: number) => void
}

type ActivityItem = {
  id: string
  kind: 'progress' | 'tool' | 'status'
  text: string
  status?: string
  detail?: string
  createdAt?: string
}

type ActivityTrace = {
  items: ActivityItem[]
  runningTools: number
  completedTools: number
  failedTools: number
  sawAnswer: boolean
  sawDone: boolean
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

function compact(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value.trim()
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function truncate(value: unknown, limit = 520): string {
  const raw = compact(value).replace(/\s+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
  return raw.length > limit ? `${raw.slice(0, limit)}...` : raw
}

function getToolCallId(event: AgentEvent, index: number) {
  const payload = asRecord(event.payload)
  return String(
    payload.toolCallId ||
      payload.tool_call_id ||
      payload.id ||
      payload.tool_call_record_id ||
      event.id ||
      `tool-${index}`,
  )
}

function getToolName(event: AgentEvent) {
  const payload = asRecord(event.payload)
  return String(payload.toolName || payload.tool_name || payload.name || event.node_name || 'tool')
}

function pushItem(items: ActivityItem[], item: ActivityItem) {
  const textValue = item.text.trim()
  if (!textValue) return
  const previous = items[items.length - 1]
  if (previous?.kind === item.kind && previous.text === textValue && previous.status === item.status) return
  items.push({ ...item, text: textValue })
}

function toolArgs(payload: UnknownRecord): UnknownRecord {
  return asRecord(payload.argsPreview || payload.args_preview || payload.tool_args)
}

function parsedToolOutput(payload: UnknownRecord): UnknownRecord {
  const raw = payload.outputPreview || payload.output_preview
  if (typeof raw === 'string') {
    try {
      return asRecord(JSON.parse(raw))
    } catch {
      return {}
    }
  }
  return asRecord(raw)
}

function webSearchDetail(payload: UnknownRecord): string {
  const rounds = Array.isArray(payload.search_rounds) ? payload.search_rounds : []
  const roundText = rounds
    .slice(0, 2)
    .map((item) => {
      const record = asRecord(item)
      const round = String(record.round || '')
      const query = String(record.query || '')
      const count = String(record.result_count ?? '')
      const observation = String(record.observation || '')
      return [`Round ${round}`, query, count ? `${count} results` : '', observation].filter(Boolean).join(' · ')
    })
    .join('\n')
  const previewResults = Array.isArray(payload.results_preview) ? payload.results_preview : []
  if (previewResults.length) {
    const resultsText = previewResults
      .slice(0, 5)
      .map((item, index) => {
        const record = asRecord(item)
        const title = String(record.title || record.url || `Result ${index + 1}`)
        const url = String(record.url || '')
        const snippet = String(record.snippet || '').trim()
        return [title, url, snippet].filter(Boolean).join('\n')
      })
      .join('\n\n')
    return [String(payload.reasoning_summary || '').trim(), roundText, resultsText].filter(Boolean).join('\n\n')
  }
  const parsed = parsedToolOutput(payload)
  const output = asRecord(parsed.output || parsed)
  const results = Array.isArray(output.results) ? output.results : []
  if (!results.length) return [roundText, String(output.error || parsed.error || payload.error || '').trim()].filter(Boolean).join('\n\n')
  return results
    .slice(0, 5)
    .map((item, index) => {
      const record = asRecord(item)
      const title = String(record.title || record.url || `Result ${index + 1}`)
      const url = String(record.url || '')
      const snippet = String(record.snippet || '').trim()
      return [title, url, snippet].filter(Boolean).join('\n')
    })
    .join('\n\n')
}

function webSearchResultCount(payload: UnknownRecord) {
  const directCount = payload.resultCount ?? payload.result_count
  if (typeof directCount === 'number') return directCount
  if (typeof directCount === 'string' && directCount.trim() !== '') return Number(directCount) || 0
  const parsed = parsedToolOutput(payload)
  const output = asRecord(parsed.output || parsed)
  const results = Array.isArray(output.results) ? output.results : []
  return results.length
}

function webSearchError(payload: UnknownRecord) {
  if (payload.error) return String(payload.error).trim()
  const parsed = parsedToolOutput(payload)
  const output = asRecord(parsed.output || parsed)
  return String(output.error || parsed.error || '').trim()
}

function isLocalSystemTool(toolName: string) {
  return toolName.startsWith('system.')
}

function toolStartedText(locale: 'en' | 'zh', toolName: string, payload: UnknownRecord) {
  if (toolName === 'web.search') {
    const query = String(toolArgs(payload).query || '').trim()
    return query ? text(locale, `正在联网搜索：${query}`, `Searching the web: ${query}`) : text(locale, '正在联网搜索', 'Searching the web')
  }
  if (isLocalSystemTool(toolName)) {
    return text(locale, `正在调用本地工具：${toolName}`, `Calling local tool: ${toolName}`)
  }
  return text(locale, `正在调用 ${toolName}`, `Calling ${toolName}`)
}

function toolCompletedText(locale: 'en' | 'zh', toolName: string, payload: UnknownRecord) {
  if (toolName === 'web.search') {
    const count = webSearchResultCount(payload)
    return text(locale, `已读取 ${count} 条搜索结果`, `Read ${count} search result${count === 1 ? '' : 's'}`)
  }
  if (isLocalSystemTool(toolName)) {
    return text(locale, '已完成本地工具调用', 'Local tool call completed')
  }
  return text(locale, `已完成 ${toolName}`, `Finished ${toolName}`)
}

function toolFailedText(locale: 'en' | 'zh', toolName: string) {
  if (toolName === 'web.search') {
    return text(locale, '联网搜索失败，改用已有上下文回答', 'Web search failed. Using the available context.')
  }
  if (isLocalSystemTool(toolName)) {
    return text(locale, '本地工具调用失败', 'Local tool call failed')
  }
  return text(locale, `${toolName} 执行失败`, `${toolName} failed`)
}

function collectActivityTrace(message: AgentThoughtStreamProps['message'], locale: AgentThoughtStreamProps['locale']): ActivityTrace {
  const items: ActivityItem[] = []
  const thoughtIndex = new Map<string, number>()
  const toolIndex = new Map<string, number>()
  let completedTools = 0
  let failedTools = 0
  let sawAnswer = false
  let sawDone = false

  ;(message.trace_events || []).forEach((event, index) => {
    const eventType = String(event.event_type || '')
    const payload = asRecord(event.payload)
    const createdAt = String(event.created_at || payload.created_at || '')

    if (eventType === 'visible_thought_delta' || eventType === 'visible_progress_delta') {
      const id = String(payload.id || `thought-${index}`)
      const currentIndex = thoughtIndex.get(id)
      if (currentIndex === undefined) {
        thoughtIndex.set(id, items.length)
        pushItem(items, {
          id,
          kind: 'progress',
          text: String(payload.full_text || payload.text || ''),
          status: String(payload.status || 'streaming'),
          createdAt,
        })
      } else {
        const current = items[currentIndex]
        if (!current) return
        const nextText = payload.full_text ? String(payload.full_text) : `${current.text}${String(payload.text || '')}`
        items[currentIndex] = {
          ...current,
          text: nextText.trim(),
          status: String(payload.status || current.status || 'streaming'),
          createdAt: createdAt || current.createdAt,
        }
      }
      return
    }

    if (eventType === 'visible_thought') {
      pushItem(items, {
        id: `thought-${event.id || index}`,
        kind: 'progress',
        text: String(payload.text || payload.summary || ''),
        status: String(payload.status || 'completed'),
        createdAt,
      })
      return
    }

    if (eventType === 'status_step') {
      const p = asRecord(event.payload)
      const stepKey = String(p.key || event.node_name || '')
      const stepTitle = String(p.title || stepKey)
      const stepThought = String(p.thought || '')
      const stepDetail = String(p.detail || '')
      const text = stepThought ? `${stepTitle}：${stepThought}` : stepTitle
      pushItem(items, {
        id: `step-${stepKey}-${index}`,
        kind: 'progress',
        text,
        status: String(p.status || 'completed'),
        detail: stepDetail || undefined,
        createdAt: String(event.created_at || p.started_at || ''),
      })
      return
    }

    if (eventType === 'tool_call_started') {
      const id = getToolCallId(event, index)
      const toolName = getToolName(event)
      const detail = toolName === 'web.search'
        ? truncate(String(toolArgs(payload).query || ''), 360)
        : truncate(payload.argsPreview || payload.args_preview || payload.tool_args, 360)
      toolIndex.set(id, items.length)
      pushItem(items, {
        id: `tool-${id}`,
        kind: 'tool',
        text: toolStartedText(locale, toolName, payload),
        status: 'running',
        detail,
        createdAt,
      })
      return
    }

    if (eventType === 'tool_call_completed' || eventType === 'tool_call_failed') {
      const id = getToolCallId(event, index)
      const toolName = getToolName(event)
      const failed = eventType === 'tool_call_failed' || (toolName === 'web.search' && webSearchResultCount(payload) === 0 && Boolean(webSearchError(payload)))
      const detail = toolName === 'web.search'
        ? truncate(webSearchDetail(payload) || payload.error, 700)
        : truncate(payload.outputPreview || payload.output_preview || payload.error, 420)
      const currentIndex = toolIndex.get(id)
      const nextItem: ActivityItem = {
        id: `tool-${id}`,
        kind: 'tool',
        text: failed ? toolFailedText(locale, toolName) : toolCompletedText(locale, toolName, payload),
        status: failed ? 'failed' : 'completed',
        detail,
        createdAt,
      }
      if (currentIndex === undefined) {
        pushItem(items, nextItem)
      } else {
        const current = items[currentIndex]
        if (current) {
          items[currentIndex] = { ...current, ...nextItem, detail: detail || current.detail }
        } else {
          pushItem(items, nextItem)
        }
      }
      if (failed) failedTools += 1
      else completedTools += 1
      return
    }

    if (eventType === 'approval_required') {
      pushItem(items, {
        id: `approval-${event.id || index}`,
        kind: 'status',
        text: text(locale, `等待你确认工具调用：${getToolName(event)}`, `Waiting for approval: ${getToolName(event)}`),
        status: 'waiting_approval',
        createdAt,
      })
      return
    }

    if (eventType === 'approval_granted' || eventType === 'approval_approved') {
      pushItem(items, {
        id: `approval-granted-${event.id || index}`,
        kind: 'status',
        text: text(locale, '已批准，继续执行。', 'Approved. Continuing.'),
        status: 'completed',
        createdAt,
      })
      return
    }

    if (eventType === 'approval_rejected') {
      pushItem(items, {
        id: `approval-rejected-${event.id || index}`,
        kind: 'status',
        text: text(locale, '已取消，没有执行该操作。', 'Rejected. The action was not run.'),
        status: 'failed',
        createdAt,
      })
      return
    }

    if (eventType === 'answer_started') {
      pushItem(items, {
        id: `answer-started-${event.id || index}`,
        kind: 'progress',
        text: text(locale, '正在整理最终回答。', 'Preparing the final answer.'),
        status: 'running',
        createdAt,
      })
      sawAnswer = true
      return
    }

    if (eventType === 'answer_delta') {
      sawAnswer = true
      return
    }

    if (eventType === 'answer_completed' || eventType === 'run_completed') {
      sawDone = true
      return
    }

    if (eventType === 'run_failed') {
      pushItem(items, {
        id: `run-failed-${event.id || index}`,
        kind: 'status',
        text: text(locale, '任务执行失败。', 'The run failed.'),
        status: 'failed',
        detail: truncate(payload.error || payload.answer, 360),
        createdAt,
      })
    }
  })

  if (sawDone) {
    items.forEach((item) => {
      if (item.status === 'running' || item.status === 'streaming') {
        item.status = 'completed'
      }
    })
  }

  const runningTools = items.filter((item) => item.kind === 'tool' && item.status === 'running').length
  return { items, runningTools, completedTools, failedTools, sawAnswer, sawDone }
}

function approvalFrom(
  message: AgentThoughtStreamProps['message'],
  locale: AgentThoughtStreamProps['locale'],
): { approvalId: number; cardData: ApprovalCardData } {
  const events = message.trace_events || []
  let approvalStatus = 'pending'
  for (let i = events.length - 1; i >= 0; i--) {
    const et = events[i]?.event_type
    if (et === 'approval_granted' || et === 'approval_approved') { approvalStatus = 'approved'; break }
    if (et === 'approval_rejected') { approvalStatus = 'rejected'; break }
    if (et === 'tool_call_failed') { approvalStatus = 'approved'; break }
    if (et === 'tool_call_completed') { approvalStatus = 'approved'; break }
    if (et === 'run_completed') { approvalStatus = 'completed'; break }
  }
  if (message.status === 'completed') approvalStatus = 'completed'

  const event = events.find((item) => item.event_type === 'approval_required')
  const meta = asRecord(message.metadata)
  const payload = event ? asRecord(event.payload) : asRecord(meta.approval_payload)
  if (!payload.approval_id && !payload.tool_name) return { approvalId: 0, cardData: {} }

  const approvalId = Number(payload.approval_id || 0)
  return {
    approvalId,
    cardData: {
      approval_id: payload.approval_id as string | number | undefined,
      run_id: payload.run_id as string | number | undefined,
      risk_level: String(payload.risk_level || 'L3'),
      tool_name: String(payload.tool_name || ''),
      title: String(payload.title || text(locale, '需要你确认', 'Approval required')),
      preview: asRecord(payload.preview),
      tool_args: asRecord(payload.tool_args),
      safety_notes: Array.isArray(payload.safety_notes) ? payload.safety_notes : [],
      actions: Array.isArray(payload.actions) ? payload.actions : ['approve', 'reject'],
      status: approvalStatus,
    },
  }
}

function ActivityRow({ item, locale }: { item: ActivityItem; locale: 'en' | 'zh' }) {
  const [open, setOpen] = useState(false)
  const failed = item.status === 'failed'
  const running = item.status === 'running' || item.status === 'streaming' || item.status === 'waiting_approval'
  const mark = item.kind === 'tool' ? '⌘' : failed ? '!' : running ? '' : '✓'
  const canInspect = Boolean(item.detail)

  return (
    <div className={failed ? 'activity-row failed' : running ? 'activity-row running' : 'activity-row'}>
      <span className="activity-row-mark">{mark}</span>
      <div className="activity-row-body">
        <p>{item.text}</p>
        {canInspect ? (
          <button className="activity-row-detail-toggle" type="button" onClick={() => setOpen((value) => !value)}>
            {open ? text(locale, '收起详情', 'Hide details') : text(locale, '查看详情', 'View details')}
          </button>
        ) : null}
        {open && item.detail ? <pre className="activity-row-detail">{item.detail}</pre> : null}
      </div>
    </div>
  )
}

export function AgentThoughtStream({ message, locale, onApprove, onReject }: AgentThoughtStreamProps) {
  const trace = useMemo(() => collectActivityTrace(message, locale), [message, locale])
  const status = String(message.status || 'completed')
  const running = ['thinking', 'running', 'created', 'queued', 'streaming'].includes(status)
  const waitingApproval = status === 'waiting_approval'
  const resuming = status === 'resuming'
  const failed = status === 'failed'
  const done = (status === 'completed' || trace.sawDone) && !running && !waitingApproval && !resuming && !failed
  const active = running || waitingApproval || resuming
  const elapsed = seconds(message.elapsed_ms)
  const { approvalId, cardData } = approvalFrom(message, locale)
  const [open, setOpen] = useState(active || failed)

  useEffect(() => {
    setOpen(active || failed)
  }, [active, failed, message.message_id])

  if (!trace.items.length && !active && !failed && !approvalId && !trace.sawAnswer) return null

  const label = failed
    ? text(locale, '工作过程中断', 'Activity interrupted')
    : done
      ? text(locale, '已处理', 'Completed')
      : text(locale, '工作过程', 'Activity')
  const toolSummary = trace.runningTools
    ? text(locale, `正在运行 ${trace.runningTools} 个工具`, `${trace.runningTools} tool${trace.runningTools > 1 ? 's' : ''} running`)
    : trace.completedTools || trace.failedTools
      ? text(
          locale,
          `已运行 ${trace.completedTools + trace.failedTools} 条工具调用`,
          `Ran ${trace.completedTools + trace.failedTools} tool call${trace.completedTools + trace.failedTools > 1 ? 's' : ''}`,
        )
      : ''

  return (
    <div className={failed ? 'agent-thought-stream failed' : active ? 'agent-thought-stream running' : 'agent-thought-stream'}>
      <button className="thought-stream-status" type="button" onClick={() => setOpen((value) => !value)}>
        <span className={active ? 'thinking-dot active' : 'thinking-dot'} />
        <strong>
          {label}
          {elapsed ? ` ${elapsed}` : ''}
        </strong>
        {toolSummary ? <span className="activity-tool-summary">{toolSummary}</span> : null}
        <span className="thought-stream-chevron">{open ? '⌄' : '›'}</span>
      </button>

      {open ? (
        <div className="activity-timeline">
          {trace.items.length ? (
            trace.items.map((item) => <ActivityRow key={item.id} item={item} locale={locale} />)
          ) : active ? (
            <div className="activity-row running">
              <span className="activity-row-mark" />
              <div className="activity-row-body">
                <p>{text(locale, '正在理解需求。', 'Understanding the request.')}</p>
              </div>
            </div>
          ) : null}

          {approvalId ? (
            <div className="activity-approval">
              <ApprovalCard
                data={cardData}
                locale={locale}
                onApprove={onApprove}
                onReject={onReject}
              />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
