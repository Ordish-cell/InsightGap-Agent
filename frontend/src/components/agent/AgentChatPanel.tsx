import { FormEvent, useEffect, useRef, useState } from 'react'

import type { AgentChatMessage, AgentEvent, AgentRun, AgentRunStep, ChatAttachment, UnknownRecord } from '../../api/types'
import * as agent from '../../api/agent'
import { fetchDocumentBlobUrl, toApiUrl, uploadChatAttachment } from '../../api/documents'
import { JsonBlock } from '../common/JsonBlock'
import { MarkdownRenderer } from '../common/MarkdownRenderer'
import { StatusPill } from '../common/StatusPill'
import { ModelSelector } from '../llm/ModelSelector'
import { AgentThoughtStream } from './AgentThoughtStream'

type LocalChatAttachment = ChatAttachment & {
  localId: string
  file?: File
  uploadProgress: number
  uploadStatus: 'queued' | 'uploading' | 'uploaded' | 'failed'
  localPreviewUrl?: string
  error?: string
}

type AgentChatPanelProps = {
  source?: string
  pageContext?: Record<string, unknown>
  placeholder?: string
  initialTitle?: string
  debug?: boolean
  locale?: 'en' | 'zh'
}

type UiMessage = AgentChatMessage & {
  local_id?: string
  trace_events?: AgentEvent[]
}

const LIVE_TRACE_LIMIT = 4000

const zh = {
  started: '\u5f00\u59cb\u5904\u7406',
  understanding: '\u7406\u89e3\u8bf7\u6c42',
  intentChecked: '\u5b8c\u6210\u610f\u56fe\u8bc6\u522b',
  runningStep: '\u6267\u884c\u6b65\u9aa4',
  stepCompleted: '\u5b8c\u6210\u6b65\u9aa4',
  stepFailed: '\u6b65\u9aa4\u5931\u8d25',
  approvalRequired: '\u7b49\u5f85\u5ba1\u6279',
  finalCreated: '\u751f\u6210\u6700\u7ec8\u56de\u7b54',
  runCompleted: '\u6267\u884c\u5b8c\u6210',
  runFailed: '\u6267\u884c\u5931\u8d25',
  permission: '\u68c0\u67e5\u98ce\u9669',
  need: '\u5224\u65ad\u9700\u6c42',
  plan: '\u751f\u6210\u8ba1\u5212',
  context: '\u6784\u5efa\u4e0a\u4e0b\u6587',
  skill: '\u5339\u914d Skill',
  research: '\u6267\u884c\u7814\u7a76',
  rag: '\u68c0\u7d22\u77e5\u8bc6\u5e93',
  artifact: '\u751f\u6210\u4ea7\u7269',
  tool: '\u51c6\u5907\u5de5\u5177\u52a8\u4f5c',
  memory: '\u5199\u5165\u8bb0\u5fc6',
  skillDraft: '\u6c89\u6dc0 Skill',
  evaluate: '\u8bc4\u4f30\u7ed3\u679c',
  answer: '\u6574\u7406\u56de\u7b54',
  thinking: '\u6b63\u5728\u601d\u8003',
  completedReasoning: '\u5df2\u5b8c\u6210\u601d\u8003',
  answered: '\u5df2\u56de\u7b54',
  waitingRecords: '\u7b49\u5f85\u72b6\u6001\u8bb0\u5f55',
  recordsAppear: '\u8fd0\u884c\u5f00\u59cb\u540e\u4f1a\u6301\u7eed\u8ffd\u52a0\u53ef\u8bfb\u6b65\u9aa4\u3002',
  action: '\u52a8\u4f5c',
  observation: '\u89c2\u5bdf',
  next: '\u4e0b\u4e00\u6b65',
  approve: '\u6279\u51c6\u6267\u884c',
  reject: '\u62d2\u7edd',
  processing: '\u6b63\u5728\u5904\u7406...',
  noAnswer: '\u6ca1\u6709\u53ef\u663e\u793a\u7684\u56de\u7b54\u3002',
  approvalFailed: '\u5ba1\u6279\u64cd\u4f5c\u5931\u8d25\u3002',
  agentFailed: 'Agent \u8fd0\u884c\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\u3002',
  using: '\u5df2\u5e26\u5165\u4fe1\u606f\uff1a',
  tools: 'Agent \u5de5\u5177',
  send: '\u53d1\u9001',
}

function text(locale: 'en' | 'zh', zhText: string, en: string) {
  return locale === 'zh' ? zhText : en
}

function asRecord(value: unknown): UnknownRecord {
  return value && typeof value === 'object' ? (value as UnknownRecord) : {}
}

function seconds(value?: number | null) {
  if (!value) return ''
  if (value < 1000) return `${Math.max(0.1, value / 1000).toFixed(1)}s`
  return `${(value / 1000).toFixed(value > 10000 ? 0 : 1)}s`
}

function messageKey(message: UiMessage) {
  return message.local_id || message.message_id || `${message.role}-${message.id || message.created_at}`
}

function getUserVisibleMessageContent(message: { content?: unknown }): string {
  const content = (message as Record<string, unknown>).content

  if (typeof content === 'string') {
    const trimmed = content.trim()
    // Detect JSON blobs that look like internal payloads
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[{') && trimmed.endsWith('}]'))) {
      try {
        const parsed = JSON.parse(trimmed)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          // This is an internal payload dict — extract the user-facing field
          const extracted =
            (parsed as Record<string, unknown>).final_output ||
            (parsed as Record<string, unknown>).answer ||
            (parsed as Record<string, unknown>).content ||
            (parsed as Record<string, unknown>).message ||
            (parsed as Record<string, unknown>).text
          if (typeof extracted === 'string' && extracted.trim()) return extracted
          // Fall through: looks like internal JSON with no user text → suppress
          if ((parsed as Record<string, unknown>).status && ((parsed as Record<string, unknown>).artifacts || (parsed as Record<string, unknown>).route)) {
            return ''
          }
        }
      } catch {
        // Not valid JSON — return as-is
      }
    }
    return content
  }

  if (content && typeof content === 'object') {
    const obj = content as Record<string, unknown>
    const extracted =
      obj.final_output || obj.answer || obj.content || obj.message || obj.text
    if (typeof extracted === 'string' && extracted.trim()) return extracted
    return ''
  }

  return String(content ?? '')
}

function looksLikeInternalJson(text: string): boolean {
  const s = text.trimStart()
  if (!s.startsWith('{')) return false
  const head = s.slice(0, 600)
  return [
    '"status"', '"final_output"', '"artifacts"',
    '"memory_updates"', '"skill_drafts"', '"evidence"',
    '"memory_writes"', '"agent_outputs"',
  ].some(k => head.includes(k))
}

const _approvalPlaceholderPrefixes = [
  'Approval required:',
  'approval required:',
  'Approval required（',
  'approval required（',
  'Approval required (',
]

function isApprovalPlaceholder(text: string): boolean {
  if (!text) return false
  const s = text.trim()
  if (_approvalPlaceholderPrefixes.some(p => s.startsWith(p))) return true
  if (s.startsWith('⏸') && s.includes('正在等待你的审批')) return true
  return false
}

function responseMessages(response: AgentRun, fallbackUser: UiMessage, fallbackAssistant: UiMessage) {
  const answer = agent.extractRunAnswer(response)
  const runId = response.run_id || response.id || fallbackAssistant.run_id || null
  const langgraphstatus = asRecord(response.langgraphstatus)
  const steps = Array.isArray(langgraphstatus.steps) ? (langgraphstatus.steps as AgentRunStep[]) : []
  const userMessage: UiMessage = response.user_message
    ? { ...response.user_message }
    : { ...fallbackUser, run_id: runId, conversation_id: response.conversation_id || fallbackUser.conversation_id }
  const assistantMessage: UiMessage = response.assistant_message
    ? { ...response.assistant_message, content: response.assistant_message.content || answer }
    : {
        ...fallbackAssistant,
        run_id: runId,
        conversation_id: response.conversation_id || fallbackAssistant.conversation_id,
        thread_id: response.thread_id || fallbackAssistant.thread_id,
        status: response.status || 'completed',
        elapsed_ms: response.elapsed_ms,
        content: answer,
        langgraphstatus,
        steps,
        metadata: { final_response: response.final_response || response.final_payload || {} },
      }
  assistantMessage.content = assistantMessage.content || answer
  assistantMessage.status = assistantMessage.status || response.status || 'completed'
  assistantMessage.elapsed_ms = assistantMessage.elapsed_ms ?? response.elapsed_ms
  assistantMessage.langgraphstatus = assistantMessage.langgraphstatus || langgraphstatus
  assistantMessage.steps = assistantMessage.steps?.length ? assistantMessage.steps : steps
  return { userMessage, assistantMessage }
}

function parseEvent(raw: string): AgentEvent | null {
  try {
    const parsed = JSON.parse(raw) as AgentEvent | { data?: AgentEvent }
    if ('payload' in parsed || 'event_type' in parsed) return parsed as AgentEvent
    const nested = (parsed as { data?: unknown }).data
    return nested && typeof nested === 'object' ? (nested as AgentEvent) : null
  } catch {
    return null
  }
}

function eventIdentity(event: AgentEvent) {
  const sequence = event.event_seq ?? event.id
  if (sequence !== undefined && sequence !== null) return `ledger:${sequence}`
  const payload = asRecord(event.payload)
  const domainId = payload.event_id || payload.tool_call_id || payload.toolCallId || payload.approval_id
  if (domainId) return `domain:${event.event_type || ''}:${String(domainId)}`
  return ''
}

function appendTraceEvent(events: AgentEvent[] | undefined, event: AgentEvent) {
  const current = events || []
  const identity = eventIdentity(event)
  if (identity && current.some((item) => eventIdentity(item) === identity)) return current
  return [...current, event].slice(-LIVE_TRACE_LIMIT)
}

async function loadRunReplay(runId: number) {
  return agent.agentLedgerClient.replay(runId)
}

function eventTitle(event: AgentEvent, locale: 'en' | 'zh') {
  const payload = event.payload || {}
  const title = payload.title || payload.summary
  if (title) return String(title)
  const node = String(event.node_name || event.event_type || '')
  const zhMap: Record<string, string> = {
    run_started: zh.started,
    home_intent_started: zh.understanding,
    home_intent_completed: zh.intentChecked,
    node_started: zh.runningStep,
    node_completed: zh.stepCompleted,
    node_failed: zh.stepFailed,
    approval_required: zh.approvalRequired,
    final_response_created: zh.finalCreated,
    run_completed: zh.runCompleted,
    run_failed: zh.runFailed,
    permission_guard: zh.permission,
    home_intent_react: zh.need,
    planner: zh.plan,
    context_builder: zh.context,
    skill_matcher: zh.skill,
    research_agent: zh.research,
    rag_agent: zh.rag,
    artifact_agent: zh.artifact,
    tool_agent: zh.tool,
    memory_agent: zh.memory,
    skill_agent: zh.skillDraft,
    evaluator: zh.evaluate,
    final_response: zh.answer,
  }
  const en: Record<string, string> = {
    run_started: 'Started request',
    home_intent_started: 'Understanding request',
    home_intent_completed: 'Intent checked',
    node_started: 'Running step',
    node_completed: 'Step completed',
    node_failed: 'Step failed',
    approval_required: 'Approval required',
    final_response_created: 'Final answer created',
    run_completed: 'Run completed',
    run_failed: 'Run failed',
  }
  const dict = locale === 'zh' ? zhMap : en
  return dict[node] || dict[String(event.event_type || '')] || node || text(locale, zh.runningStep, 'Run event')
}

function eventThought(event: AgentEvent, locale: 'en' | 'zh') {
  const payload = event.payload || {}
  return String(payload.llm_stage_output || payload.thought || payload.summary || payload.answer || payload.final_output || payload.reason || eventTitle(event, locale))
}

function eventsToSteps(events: AgentEvent[], locale: 'en' | 'zh'): AgentRunStep[] {
  return events
    .filter((event) => !['node_started'].includes(String(event.event_type || '')))
    .slice(-12)
    .map((event, index) => {
      const payload = event.payload || {}
      const status = String(payload.status || (String(event.event_type || '').includes('failed') ? 'failed' : 'completed'))
      return {
        key: `${event.id || index}-${event.event_type || 'event'}`,
        title: eventTitle(event, locale),
        status,
        thought: eventThought(event, locale),
        action: String(payload.action || eventTitle(event, locale)),
        observation: String(payload.observation || payload.summary || ''),
        next_action: String(payload.next_action || ''),
        node_name: event.node_name,
        completed_at: event.created_at,
      }
    })
}

function stepsForMessage(message: UiMessage, locale: 'en' | 'zh') {
  return eventsToSteps(message.trace_events || [], locale)
}

function stepThought(step: AgentRunStep) {
  return String(step.llm_stage_output || step.thought || step.summary || step.detail || step.status || '')
}

function StepMeta({ label, value }: { label: string; value: unknown }) {
  const content = String(value || '').trim()
  if (!content) return null
  return (
    <div className="run-step-meta-row">
      <span>{label}</span>
      <p>{content}</p>
    </div>
  )
}

function AgentRunTraceBlock({
  message,
  locale,
  onApprove,
  onReject,
}: {
  message: UiMessage
  locale: 'en' | 'zh'
  onApprove: (approvalId: number) => void
  onReject: (approvalId: number) => void
}) {
  const steps = stepsForMessage(message, locale)
  const status = String(message.status || 'completed')
  const running = ['thinking', 'running', 'created', 'queued'].includes(status)
  const failed = status === 'failed'
  const [open, setOpen] = useState(running || failed)
  const compact = !running && !failed && steps.length <= 1 && !(message.trace_events || []).length
  const elapsed = seconds(message.elapsed_ms)
  const label = running
    ? text(locale, zh.thinking, 'Thinking')
    : failed
      ? text(locale, zh.runFailed, 'Run failed')
      : compact
        ? text(locale, zh.answered, 'Answered')
        : text(locale, zh.completedReasoning, 'Completed reasoning')
  const approvalEvent = (message.trace_events || []).find((event) => event.event_type === 'approval_required')
  const approvalPayload = approvalEvent?.payload || {}
  const approvalId = Number(approvalPayload.approval_id || asRecord(approvalPayload.approval_payload).approval_id || 0)

  useEffect(() => {
    if (running || failed) setOpen(true)
    else setOpen(false)
  }, [failed, running, message.message_id])

  if (!running && !failed && !steps.length && !(message.trace_events || []).length) return null

  return (
    <div className={failed ? 'run-trace failed' : running ? 'run-trace running' : 'run-trace'}>
      <button className="run-trace-summary" type="button" onClick={() => setOpen((value) => !value)} disabled={compact && !failed}>
        <span className={running ? 'thinking-dot active' : 'thinking-dot'} />
        <strong>
          {label}
          {elapsed ? ` - ${elapsed}` : ''}
        </strong>
        {!compact || failed ? <span className="run-trace-chevron">{open ? 'v' : '>'}</span> : null}
      </button>
      {open && (!compact || failed) ? (
        <div className="run-step-list">
          {steps.length ? (
            steps.map((step, index) => (
              <div className={`run-step ${step.status || 'completed'}`} key={`${step.key || step.title || 'step'}-${index}`}>
                <span className="run-step-mark">{step.status === 'failed' ? '!' : step.status === 'running' ? '.' : 'ok'}</span>
                <div className="run-step-body">
                  <strong>{String(step.title || step.node_name || text(locale, zh.runningStep, 'Run step'))}</strong>
                  {stepThought(step) ? <p className="run-step-thought">{stepThought(step)}</p> : null}
                  <div className="run-step-meta">
                    <StepMeta label={text(locale, zh.action, 'Action')} value={step.action} />
                    <StepMeta label={text(locale, zh.observation, 'Observation')} value={step.observation} />
                    <StepMeta label={text(locale, zh.next, 'Next')} value={step.next_action} />
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="run-step running">
              <span className="run-step-mark">.</span>
              <div>
                <strong>{text(locale, zh.waitingRecords, 'Waiting for status records')}</strong>
                <p>{text(locale, zh.recordsAppear, 'Readable steps appear as the run progresses.')}</p>
              </div>
            </div>
          )}
          {approvalId ? (
            <div className="approval-inline-actions">
              <small>{String(approvalPayload.risk_level || approvalPayload.permission_level || 'L3')}</small>
              <button className="light-mini-button" type="button" onClick={() => onApprove(approvalId)}>
                {text(locale, zh.approve, 'Approve')}
              </button>
              <button className="light-mini-button" type="button" onClick={() => onReject(approvalId)}>
                {text(locale, zh.reject, 'Reject')}
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function AuthenticatedImage({
  documentId,
  alt,
  className,
  fallbackText,
}: {
  documentId: number
  alt: string
  className?: string
  fallbackText?: string
}) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [error, setError] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(false)

    fetchDocumentBlobUrl(documentId)
      .then((url) => {
        if (!cancelled) {
          setBlobUrl(url)
          setLoading(false)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError(true)
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
      if (blobUrl) URL.revokeObjectURL(blobUrl)
    }
  }, [documentId])

  if (loading) {
    return <div className={`${className || ''} authenticated-image-placeholder`}>⏳</div>
  }
  if (error) {
    return <div className={`${className || ''} authenticated-image-error`}>{fallbackText || alt || '图片加载失败'}</div>
  }
  if (blobUrl) {
    return <img src={blobUrl} alt={alt} className={className} />
  }
  return null
}

function AgentMessageItem({
  message,
  locale,
  debug,
  onApprove,
  onReject,
}: {
  message: UiMessage
  locale: 'en' | 'zh'
  debug: boolean
  onApprove: (approvalId: number) => void
  onReject: (approvalId: number) => void
}) {
  const visibleContent = getUserVisibleMessageContent(message)

  const messageAttachments: ChatAttachment[] =
    message.attachments ||
    ((message.metadata as UnknownRecord)?.attachments as ChatAttachment[]) ||
    []

  if (message.role === 'user') {
    return (
      <article className="chat-message user">
        <div className="message-bubble">
          {(visibleContent || message.content) || null}
          {messageAttachments.length > 0 ? (
            <div className="message-attachments">
              {messageAttachments.map((item) => (
                <a
                  key={item.document_id}
                  className={
                    item.kind === 'image'
                      ? 'message-attachment image'
                      : 'message-attachment file'
                  }
                  href={toApiUrl(item.preview_url)}
                  target="_blank"
                  rel="noreferrer"
                >
                  {item.kind === 'image' ? (
                    <AuthenticatedImage
                      documentId={item.document_id}
                      alt={item.filename}
                      className="message-attachment-image"
                      fallbackText={item.filename}
                    />
                  ) : (
                    <div className="message-attachment-file">
                      <span className="message-attachment-file-icon">📄</span>
                      <span className="message-attachment-file-name">{item.filename}</span>
                    </div>
                  )}
                </a>
              ))}
            </div>
          ) : null}
        </div>
      </article>
    )
  }

  return (
    <article className="chat-message assistant">
      <div className="assistant-run-message">
        {debug ? (
          <AgentRunTraceBlock message={message} locale={locale} onApprove={onApprove} onReject={onReject} />
        ) : (
          <AgentThoughtStream message={message} locale={locale} onApprove={onApprove} onReject={onReject} />
        )}
        <div className="message-bubble answer-content">
          {visibleContent && !isApprovalPlaceholder(visibleContent) ? (
            <MarkdownRenderer content={visibleContent} />
          ) : message.status === 'thinking' || message.status === 'streaming' || message.status === 'created' || message.status === 'running' || message.status === 'waiting_approval' ? null : (
            text(locale, zh.noAnswer, 'No answer to display.')
          )}
        </div>
      </div>
    </article>
  )
}

export function AgentChatPanel({
  source = 'agent_page',
  pageContext = {},
  placeholder = 'Enter a task for the Agent',
  initialTitle = 'What should we build or research?',
  debug = false,
  locale = 'en',
}: AgentChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const streamRef = useRef<{ close: () => void } | null>(null)
  const [userInput, setUserInput] = useState('')
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [activeConversationId, setActiveConversationId] = useState('')
  const [currentRun, setCurrentRun] = useState<AgentRun | null>(null)
  const [running, setRunning] = useState(false)
  const [selectedModelConfigId, setSelectedModelConfigId] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [replayLoading, setReplayLoading] = useState(false)
  const [replayError, setReplayError] = useState('')

  const [attachments, setAttachments] = useState<LocalChatAttachment[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const lastPasteAtRef = useRef(0)
  const [isComposing, setIsComposing] = useState(false)

  const hasConversation = messages.length > 0 || running || replayLoading || Boolean(activeConversationId) || Boolean(error) || Boolean(replayError)
  const selectedFeedTitle = String(pageContext.selected_feed_card_title || '')

  function handleNetworkStatus(status: 'recovering' | 'caught_up' | 'retrying') {
    if (status === 'caught_up') {
      setReplayError('')
      return
    }
    setReplayError(status === 'recovering'
      ? text(locale, '正在恢复连接…', 'Recovering connection…')
      : text(locale, '恢复失败，正在继续重试…', 'Recovery failed; continuing to retry…'))
  }

  // ── attachment helpers ──
  const buildScreenshotFilename = (mimeType: string) => {
    const now = new Date()
    const pad = (value: number) => String(value).padStart(2, '0')
    const ext = mimeType === 'image/jpeg' ? 'jpg' : mimeType === 'image/webp' ? 'webp' : 'png'
    return [
      'screenshot',
      now.getFullYear(),
      pad(now.getMonth() + 1),
      pad(now.getDate()),
      `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`,
    ].join('-') + `.${ext}`
  }

  const removeAttachment = (localId: string) => {
    setAttachments((prev) => {
      const target = prev.find((item) => item.localId === localId)
      if (target?.localPreviewUrl) {
        URL.revokeObjectURL(target.localPreviewUrl)
      }
      return prev.filter((item) => item.localId !== localId)
    })
  }

  const handleFilesSelected = async (files: File[]) => {
    for (const file of files) {
      const localId = crypto.randomUUID()
      const isImage = file.type.startsWith('image/')
      const localPreviewUrl = isImage ? URL.createObjectURL(file) : undefined

      const initialAttachment: LocalChatAttachment = {
        localId,
        file,
        document_id: -1,
        filename: file.name,
        file_type: file.name.split('.').pop() || '',
        mime_type: file.type,
        kind: isImage ? 'image' : 'document',
        size: file.size,
        preview_url: localPreviewUrl,
        localPreviewUrl,
        uploadProgress: 0,
        uploadStatus: 'uploading',
        status: 'uploading',
      }

      setAttachments((prev) => [...prev, initialAttachment])

      uploadChatAttachment(file, (progress) => {
        setAttachments((prev) =>
          prev.map((item) =>
            item.localId === localId ? { ...item, uploadProgress: progress } : item,
          ),
        )
      })
        .then((uploaded) => {
          const isReady = uploaded.status === 'ready' && (uploaded.kind === 'image' || (((uploaded as unknown) as Record<string, unknown>).chunks_count as number ?? 0) > 0)
          setAttachments((prev) =>
            prev.map((item) =>
              item.localId === localId
                ? {
                    ...item,
                    ...uploaded,
                    localId,
                    file: item.file,
                    localPreviewUrl: item.localPreviewUrl,
                    uploadProgress: 100,
                    uploadStatus: isReady ? 'uploaded' : 'failed',
                    status: isReady ? 'uploaded' : 'failed',
                    error: isReady ? undefined : (item.kind === 'document' ? '文档解析失败，请检查文件格式或重试' : undefined),
                  }
                : item,
            ),
          )
        })
        .catch((error: unknown) => {
          setAttachments((prev) =>
            prev.map((item) =>
              item.localId === localId
                ? {
                    ...item,
                    uploadStatus: 'failed',
                    status: 'failed',
                    error: error instanceof Error ? error.message : '上传失败',
                  }
                : item,
            ),
          )
        })
    }
  }

  const handlePaste = (event: React.ClipboardEvent<HTMLFormElement | HTMLTextAreaElement>) => {
    const now = Date.now()
    if (now - lastPasteAtRef.current < 100) return

    const clipboardItems = Array.from(event.clipboardData?.items || [])
    const imageFiles: File[] = []

    for (const item of clipboardItems) {
      if (!item.type.startsWith('image/')) continue
      const blob = item.getAsFile()
      if (!blob) continue

      const file = new File([blob], buildScreenshotFilename(item.type), {
        type: item.type || 'image/png',
        lastModified: now,
      })
      imageFiles.push(file)
    }

    if (imageFiles.length > 0) {
      lastPasteAtRef.current = now
      event.preventDefault()
      void handleFilesSelected(imageFiles)
    }
  }

  const handleComposerKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter') return

    if (event.shiftKey) {
      // Let Shift+Enter insert newline (default behavior)
      return
    }

    // Don't send while IME is composing
    if (isComposing || (event.nativeEvent as KeyboardEvent & { isComposing?: boolean }).isComposing) {
      return
    }

    if (!canSend) {
      event.preventDefault()
      return
    }

    event.preventDefault()
    event.currentTarget.form?.requestSubmit()
  }

  // cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      attachments.forEach((item) => {
        if (item.localPreviewUrl) URL.revokeObjectURL(item.localPreviewUrl)
      })
    }
  }, [])

  useEffect(() => {
    const pendingConversationId = sessionStorage.getItem('agentOpenConversationId')
    if (pendingConversationId) {
      sessionStorage.removeItem('agentOpenConversationId')
      void loadConversation(pendingConversationId)
    } else {
      void loadLatestConversation()
    }

    function openConversation(event: Event) {
      const detail = (event as CustomEvent<{ conversationId?: string }>).detail || {}
      if (detail.conversationId) void loadConversation(detail.conversationId)
    }

    function newConversation() {
      startNewConversation()
    }

    window.addEventListener('agent:open-conversation', openConversation as EventListener)
    window.addEventListener('agent:new-conversation', newConversation)
    return () => {
      streamRef.current?.close()
      window.removeEventListener('agent:open-conversation', openConversation as EventListener)
      window.removeEventListener('agent:new-conversation', newConversation)
    }
  }, [])

  useEffect(() => {
    if (hasConversation) bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [hasConversation, messages, running, error])

  async function loadLatestConversation() {
    try {
      const result = await agent.listConversations({ status: 'active', limit: 1 })
      const latest = result.items?.[0]
      if (latest?.conversation_id) await loadConversation(latest.conversation_id)
    } catch {
      // Home should still be usable even if the conversation menu cannot load.
    }
  }

  async function loadConversation(conversationId: string) {
    streamRef.current?.close()
    setError('')
    setReplayError('')
    setReplayLoading(true)
    try {
      const item = await agent.getConversation(conversationId)
      setActiveConversationId(item.conversation_id)
      const rememberedModelId = Number((item.metadata as UnknownRecord | undefined)?.model_config_id || 0)
      setSelectedModelConfigId(rememberedModelId > 0 ? rememberedModelId : null)
      const conversationMessages = (item.messages || []) as UiMessage[]
      const runIds = Array.from(new Set(
        conversationMessages
          .filter((message) => message.role === 'assistant' && Number(message.run_id) > 0)
          .map((message) => Number(message.run_id)),
      ))
      const replayResults = await Promise.allSettled(runIds.map(async (runId) => [runId, await loadRunReplay(runId)] as const))
      const replayByRun = new Map<number, AgentEvent[]>()
      let failedRuns = 0
      replayResults.forEach((result) => {
        if (result.status === 'fulfilled') replayByRun.set(result.value[0], result.value[1])
        else failedRuns += 1
      })
      const restoredMessages = conversationMessages.map((message) => {
        const replay = replayByRun.get(Number(message.run_id))
        return replay ? { ...message, trace_events: replay } : message
      })
      setMessages(restoredMessages)
      const activeMessage = [...restoredMessages].reverse().find((message) =>
        message.role === 'assistant' &&
        Number(message.run_id) > 0 &&
        ['created', 'thinking', 'running', 'streaming', 'resuming'].includes(String(message.status)),
      )
      if (activeMessage?.run_id) {
        const activeRunId = Number(activeMessage.run_id)
        setRunning(true)
        streamRef.current = agent.agentLedgerClient.tailRun(activeRunId, 0, {
          onNetworkStatus: handleNetworkStatus,
          onMessage: (message) => {
            const parsed = parseEvent(message.data)
            if (!parsed) return
            const payload = asRecord(parsed.payload)
            setMessages((items) => items.map((item) => {
              if (item.role !== 'assistant' || Number(item.run_id) !== activeRunId) return item
              const trace_events = appendTraceEvent(item.trace_events, parsed)
              if (parsed.event_type === 'answer_delta') {
                return { ...item, status: 'streaming', content: `${item.content || ''}${String(payload.text || '')}`, trace_events }
              }
              if (parsed.event_type === 'answer_completed') {
                return { ...item, status: 'completed', content: String(payload.answer || item.content || ''), trace_events }
              }
              if (parsed.event_type === 'approval_required' || parsed.event_type === 'run_paused') {
                return { ...item, status: 'waiting_approval', trace_events }
              }
              if (parsed.event_type === 'run_completed') {
                const response = asRecord(payload.response) as AgentRun
                return { ...item, status: 'completed', content: agent.extractRunAnswer(response) || String(payload.answer || item.content || ''), trace_events }
              }
              if (parsed.event_type === 'run_failed' || parsed.event_type === 'run_interrupted') {
                return { ...item, status: 'failed', content: String(payload.error || item.content || ''), trace_events }
              }
              return { ...item, trace_events }
            }))
            if (['run_completed', 'run_failed', 'run_interrupted', 'run_paused'].includes(String(parsed.event_type))) setRunning(false)
          },
        })
      }
      if (failedRuns) {
        setReplayError(text(locale, `有 ${failedRuns} 条运行记录未能恢复，可重新打开会话重试。`, `${failedRuns} run timeline(s) could not be restored. Reopen the conversation to retry.`))
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : text(locale, '会话加载失败。', 'Failed to load conversation.'))
    } finally {
      setReplayLoading(false)
    }
  }

  function startNewConversation() {
    streamRef.current?.close()
    setActiveConversationId('')
    setMessages([])
    setCurrentRun(null)
    setSelectedModelConfigId(null)
    setError('')
    setReplayError('')
    setReplayLoading(false)
  }

  async function handleApproval(approvalId: number, approved: boolean) {
    try {
      // 1. Call approve/reject API
      const result = (approved
        ? await agent.approveRunApproval(approvalId, { decision: 'approved' })
        : await agent.rejectRunApproval(approvalId, { decision: 'rejected' })) as Record<string, unknown>

      const runId = result?.run_id as number | undefined

      // 2. Resume the background task and reconnect to the same ledger.
      if (runId) {
        streamRef.current?.close()
        setRunning(true)

        let resumeContent = ''
        streamRef.current = agent.agentLedgerClient.resumeAndTail(runId, {
          onNetworkStatus: handleNetworkStatus,
          onMessage: (message) => {
            const parsed = parseEvent(message.data)
            if (!parsed) return
            const payload = asRecord(parsed.payload)

            // Feed events to the same assistant message
            setMessages((items) =>
              items.map((item) => {
                // Find the waiting_approval assistant message for this run
                const isTargetAssistant =
                  item.role === 'assistant' &&
                  (item.status === 'waiting_approval' ||
                   item.status === 'resuming' ||
                   item.run_id === runId)

                if (!isTargetAssistant) return item

                if (parsed.event_type === 'answer_delta') {
                  const delta = typeof payload.text === 'string' ? payload.text : String(payload.text || '')
                  // Defense: suppress stale approval_required text after approval granted
                  if (delta.includes('Run failed: approval_required') || (resumeContent + delta).includes('Run failed: approval_required')) {
                    console.warn('[AgentChatPanel resume] Suppressed answer_delta with approval_required:', delta.slice(0, 100))
                    return { ...item, status: item.status === 'waiting_approval' ? 'resuming' : item.status }
                  }
                  if (delta.includes('Approval required:') || (resumeContent + delta).includes('Approval required:')) {
                    console.warn('[AgentChatPanel resume] Suppressed answer_delta with Approval required')
                    return { ...item, status: item.status === 'waiting_approval' ? 'resuming' : item.status }
                  }
                  resumeContent += delta
                  return {
                    ...item,
                    status: 'streaming',
                    content: resumeContent,
                    trace_events: appendTraceEvent(item.trace_events, parsed),
                  }
                }

                if (parsed.event_type === 'answer_completed' || parsed.event_type === 'run_completed') {
                  const finalAnswer = typeof payload.answer === 'string' ? payload.answer : resumeContent
                  return {
                    ...item,
                    status: 'completed',
                    content: finalAnswer || resumeContent,
                    trace_events: appendTraceEvent(item.trace_events, parsed),
                  }
                }

                if (parsed.event_type === 'run_failed') {
                  const failText = String(payload.error || payload.answer || '执行失败')
                  const reason = String(payload.reason || '')

                  // approval_context_gone: session was deleted, card shows expired
                  if (reason === 'approval_context_gone') {
                    return {
                      ...item,
                      status: 'cancelled',
                      content: '该审批所属的会话或运行已经不存在，无法继续执行。',
                      trace_events: appendTraceEvent(item.trace_events, parsed),
                    }
                  }

                  // Defense: spurious approval_required run_failed after approval+tool
                  const hasApprovalGranted = (item.trace_events || []).some(
                    (e) => e.event_type === 'approval_granted' || e.event_type === 'approval_approved'
                  )
                  const hasToolEvent = (item.trace_events || []).some(
                    (e) => e.event_type === 'tool_call_completed' || e.event_type === 'tool_call_failed'
                  )
                  if (hasApprovalGranted && hasToolEvent && failText.toLowerCase().includes('approval_required')) {
                    console.warn('[AgentChatPanel] Suppressed spurious run_failed after approval+tool', failText)
                    return { ...item, status: item.status, trace_events: appendTraceEvent(item.trace_events, parsed) }
                  }

                  return {
                    ...item,
                    status: 'failed',
                    content: failText,
                    trace_events: appendTraceEvent(item.trace_events, parsed),
                  }
                }

                // approval_granted, approval_rejected, run_resumed, visible_thought_delta, tool_call_*
                return {
                  ...item,
                  status: item.status === 'waiting_approval' ? 'resuming' : item.status,
                  trace_events: appendTraceEvent(item.trace_events, parsed),
                }
              })
            )

            if (parsed.event_type === 'run_completed' || parsed.event_type === 'run_failed') {
              setRunning(false)
              streamRef.current?.close()
              streamRef.current = null
            }
          },
          onError: () => {
            setError(text(locale, zh.agentFailed, 'Agent run failed.'))
            setRunning(false)
          },
        })
      } else {
        // The approval response is authoritative even when no resume stream is needed.
        setMessages((items) =>
          items.map((message) =>
            message.role === 'assistant' && message.status === 'waiting_approval'
              ? { ...message, status: approved ? 'resuming' : 'cancelled' }
              : message
          )
        )
      }
    } catch (exc) {
      const errMsg = exc instanceof Error ? exc.message : String(exc)
      const status = (exc as { status?: number }).status

      // APPROVAL_CONTEXT_GONE (409): run/conversation/message was deleted
      if (status === 409 && errMsg.includes('APPROVAL_CONTEXT_GONE')) {
        setMessages((items) =>
          items.map((message) =>
            message.role === 'assistant' && message.status === 'waiting_approval'
              ? {
                  ...message,
                  status: 'cancelled',
                  content: '该审批所属的会话或运行已经不存在，无法继续执行。',
                }
              : message
          )
        )
        setRunning(false)
        return
      }

      // CONVERSATION_HAS_PENDING_APPROVAL (409): delete-blocked, never reaches approve
      // handler.  If it somehow does, show a toast — don't touch ApprovalCard.
      if (status === 409 && errMsg.includes('CONVERSATION_HAS_PENDING_APPROVAL')) {
        setError('当前会话有等待审批的操作，请先同意或拒绝后再删除。')
        setRunning(false)
        return
      }

      setError(errMsg || text(locale, zh.approvalFailed, 'Approval action failed.'))
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const input = userInput.trim()
    const uploadedAttachments = attachments.filter(
      (item) => item.uploadStatus === 'uploaded' && item.document_id > 0,
    )
    const hasUploading = attachments.some((item) => item.uploadStatus === 'uploading')
    if ((!input && uploadedAttachments.length === 0) || running || hasUploading) return

    streamRef.current?.close()
    setUserInput('')
    setAttachments([])
    setError('')
    setRunning(true)

    const attachmentIds = uploadedAttachments.map((item) => item.document_id)
    const attachmentSnapshot: ChatAttachment[] = uploadedAttachments.map((item) => ({
      document_id: item.document_id,
      filename: item.filename,
      file_type: item.file_type,
      mime_type: item.mime_type,
      kind: item.kind,
      size: item.size,
      preview_url: item.preview_url,
      status: item.status,
      ingest_status: item.ingest_status,
    }))

    const effectiveInput = input || (uploadedAttachments.length > 0 ? '请分析用户上传的文件。' : '')

    const now = Date.now()
    const localConversationId = activeConversationId
    const localUser: UiMessage = {
      message_id: `local-user-${now}`,
      role: 'user',
      content: effectiveInput,
      conversation_id: localConversationId,
      status: 'completed',
      local_id: `local-user-${now}`,
      attachments: attachmentSnapshot,
      metadata: { attachments: attachmentSnapshot },
    }
    const localAssistant: UiMessage = {
      message_id: `local-assistant-${now}`,
      role: 'assistant',
      content: '',
      conversation_id: localConversationId,
      status: 'thinking',
      trace_events: [
        {
          event_type: 'visible_progress_delta',
          visibility: 'user',
          display_channel: 'thinking',
          payload: {
            id: `local-start-${now}`,
            text: text(locale, '我开始处理了，会把执行过程按步骤展示在这里。', 'I started working and will show the process here step by step.'),
            status: 'streaming',
          },
          created_at: new Date(now).toISOString(),
        },
      ],
      local_id: `local-assistant-${now}`,
    }
    setMessages((items) => [...items, localUser, localAssistant])

    let liveRunId: number | undefined
    let liveAssistantMessageId = localAssistant.message_id

    streamRef.current = agent.agentLedgerClient.startAndTail(
      {
        onNetworkStatus: handleNetworkStatus,
        user_input: effectiveInput,
        input: effectiveInput,
        conversation_id: activeConversationId || undefined,
        model_config_id: selectedModelConfigId,
        source,
        page_context: pageContext,
        auto_skill: true,
        use_existing_skills: true,
        create_skill_draft_if_reusable: true,
        attachment_ids: attachmentIds,
      },
      {
        onMessage: (message) => {
          const parsed = parseEvent(message.data)
          if (!parsed) return
          const payload = asRecord(parsed.payload)

          if (parsed.event_type === 'run_created') {
            const userMessage = asRecord(payload.user_message) as UiMessage
            const assistantMessage = asRecord(payload.assistant_message) as UiMessage
            liveRunId = Number(payload.run_id || parsed.run_id || assistantMessage.run_id || 0) || undefined
            liveAssistantMessageId = assistantMessage.message_id || liveAssistantMessageId
            const nextConversationId = String(payload.conversation_id || assistantMessage.conversation_id || '')
            if (nextConversationId) setActiveConversationId(nextConversationId)
            setMessages((items) =>
              items.map((item) => {
                if (item.message_id === localUser.message_id) return { ...localUser, ...userMessage }
                if (item.message_id === localAssistant.message_id) {
                  return {
                    ...localAssistant,
                    ...assistantMessage,
                    message_id: localAssistant.message_id,
                    status: 'thinking',
                    trace_events: item.trace_events || localAssistant.trace_events || [],
                  }
                }
                return item
              })
            )
            return
          }

          if (parsed.event_type === 'run_completed') {
            const response = asRecord(payload.response) as AgentRun
            setCurrentRun(response)
            const { userMessage, assistantMessage } = responseMessages(response, localUser, localAssistant)
            setMessages((items) =>
              items.map((item) => {
                if (item.message_id === userMessage.message_id || item.message_id === localUser.message_id) return userMessage
                if (item.role === 'assistant' && (item.message_id === assistantMessage.message_id || item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)) {
                  const streamedContent = String(item.content || '')
                  const fallbackContent = String(assistantMessage.content || '')
                  // If streamed content was JSON, replace it with the clean answer
                  if (looksLikeInternalJson(streamedContent)) {
                    return {
                      ...assistantMessage,
                      content: fallbackContent || getUserVisibleMessageContent(response as unknown as { content?: unknown }),
                      trace_events: appendTraceEvent(item.trace_events, parsed),
                    }
                  }
                  // Prevent short final answer from overwriting longer streamed content
                  if (streamedContent.trim() && fallbackContent.trim() && fallbackContent.trim().length < streamedContent.trim().length * 0.5) {
                    return {
                      ...assistantMessage,
                      content: streamedContent,
                      trace_events: appendTraceEvent(item.trace_events, parsed),
                    }
                  }
                  const finalContent = streamedContent.trim() ? streamedContent : fallbackContent
                  return {
                    ...assistantMessage,
                    content: finalContent,
                    trace_events: appendTraceEvent(item.trace_events, parsed),
                  }
                }
                return item
              })
            )
            setRunning(false)
            streamRef.current?.close()
            streamRef.current = null
            return
          }

          if (parsed.event_type === 'answer_delta') {
            const rawText = payload.text
            let delta = ''
            if (typeof rawText === 'string') {
              delta = rawText
            } else if (rawText !== undefined && rawText !== null) {
              delta = String(rawText)
            }
            if (delta) {
              setMessages((items) =>
                items.map((item) => {
                  if (item.role === 'assistant' && (item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)) {
                    // Defense: suppress stale approval_required answer text after approval granted
                    const hasApprovalGranted = (item.trace_events || []).some(
                      (e: { event_type?: string }) => e.event_type === 'approval_granted' || e.event_type === 'approval_approved'
                    )
                    if (hasApprovalGranted && delta.includes('Run failed: approval_required')) {
                      console.warn('[AgentChatPanel] Suppressed answer_delta with approval_required after approval granted:', delta.slice(0, 100))
                      return { ...item, status: item.status === 'waiting_approval' ? 'resuming' : item.status }
                    }
                    if (hasApprovalGranted && delta.includes('Approval required:')) {
                      console.warn('[AgentChatPanel] Suppressed answer_delta with Approval required after approval granted')
                      return { ...item, status: item.status === 'waiting_approval' ? 'resuming' : item.status }
                    }

                    const candidateContent = `${item.content || ''}${delta}`
                    // Suppress streaming if the content starts to look like internal JSON
                    if (looksLikeInternalJson(candidateContent)) {
                      return { ...item, status: 'streaming' }
                    }
                    return {
                      ...item,
                      run_id: liveRunId || item.run_id,
                      status: 'streaming',
                      content: candidateContent,
                      trace_events: appendTraceEvent(item.trace_events, parsed),
                    }
                  }
                  return item
                })
              )
            }
            return
          }

          if (parsed.event_type === 'answer_completed') {
            const rawAnswer = payload.answer
            let completedAnswer = ''
            if (typeof rawAnswer === 'string') {
              completedAnswer = rawAnswer
            } else if (rawAnswer && typeof rawAnswer === 'object') {
              const obj = rawAnswer as Record<string, unknown>
              completedAnswer = String(obj.final_output || obj.answer || obj.content || '')
            }
            setMessages((items) =>
              items.map((item) =>
                item.role === 'assistant' && (item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)
                  ? {
                      ...item,
                      run_id: liveRunId || item.run_id,
                      status: 'completed',
                      content: completedAnswer || item.content,
                      trace_events: appendTraceEvent(item.trace_events, parsed),
                    }
                  : item
              )
            )
            return
          }

          if (parsed.event_type === 'run_failed') {
            const failedText = String(payload.error || payload.answer || text(locale, zh.agentFailed, 'Agent run failed. Please try again.'))
            setMessages((items) =>
              items.map((item) =>
                item.role === 'assistant' && (item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)
                  ? { ...item, status: 'failed', content: failedText, error_message: failedText, trace_events: appendTraceEvent(item.trace_events, parsed) }
                  : item
              )
            )
            setError(failedText)
            setRunning(false)
            return
          }

          if (parsed.event_type === 'answer_started') {
            setMessages((items) =>
              items.map((item) =>
                item.role === 'assistant' && (item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)
                  ? {
                      ...item,
                      status: item.content ? item.status : 'streaming',
                    }
                  : item
              )
            )
            return
          }

          if (parsed.event_type === 'approval_required') {
            setMessages((items) =>
              items.map((item) =>
                item.role === 'assistant' && (item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)
                  ? {
                      ...item,
                      run_id: liveRunId || item.run_id,
                      status: 'waiting_approval',
                      metadata: {
                        ...item.metadata,
                        approval_required: true,
                        approval_id: payload.approval_id,
                        approval_payload: payload,
                      },
                      trace_events: appendTraceEvent(item.trace_events, parsed),
                    }
                  : item
              )
            )
            return
          }

          if (parsed.event_type === 'run_paused') {
            setMessages((items) =>
              items.map((item) =>
                item.role === 'assistant' && (item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)
                  ? {
                      ...item,
                      status: 'waiting_approval',
                      trace_events: appendTraceEvent(item.trace_events, parsed),
                    }
                  : item
              )
            )
            // Stop the loading state — run is paused waiting for user
            setRunning(false)
            return
          }

          setMessages((items) =>
            items.map((item) =>
              item.role === 'assistant' && (item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id)
                ? {
                    ...item,
                    run_id: liveRunId || item.run_id,
                    status: (item.status === 'completed' || item.status === 'streaming' || item.status === 'waiting_approval') ? item.status : 'thinking',
                    trace_events: appendTraceEvent(item.trace_events, parsed),
                  }
                : item
            )
          )
        },
        onError: () => {
          const failedText = text(locale, zh.agentFailed, 'Agent run failed. Please try again.')
          setMessages((items) =>
            items.map((message) =>
              message.message_id === localAssistant.message_id || message.message_id === liveAssistantMessageId
                ? { ...message, status: 'failed', content: failedText, error_message: failedText }
                : message
            )
          )
          setError(failedText)
          setRunning(false)
        },
      }
    )
  }

  const uploadedAttachmentsForSend = attachments.filter(
    (item) => item.uploadStatus === 'uploaded' && item.document_id > 0,
  )
  const hasUploading = attachments.some((item) => item.uploadStatus === 'uploading')
  const hasText = userInput.trim().length > 0
  const hasUploadedForSend = uploadedAttachmentsForSend.length > 0
  const hasFailedOnly =
    attachments.length > 0 && uploadedAttachmentsForSend.length === 0 && !hasText

  const canSend =
    !running && !hasUploading && Boolean(selectedModelConfigId) && (hasText || hasUploadedForSend) && !hasFailedOnly

  const composer = (
    <form
      className={hasConversation ? 'codex-composer docked' : 'codex-composer centered'}
      onSubmit={submit}
      onPaste={handlePaste}
      onDragOver={(event) => {
        event.preventDefault()
        setDragActive(true)
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDragActive(false)
        const files = Array.from(event.dataTransfer.files || [])
        void handleFilesSelected(files)
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        hidden
        accept="image/*,.pdf,.docx,.txt,.md,.csv,.xlsx,.json,.html,.htm"
        onChange={(event) => {
          const files = Array.from(event.target.files || [])
          void handleFilesSelected(files)
          event.target.value = ''
        }}
      />

      {selectedFeedTitle ? <div className="selected-context-pill">{text(locale, `${zh.using}${selectedFeedTitle}`, `Using: ${selectedFeedTitle}`)}</div> : null}

      {attachments.length > 0 ? (
        <div className="composer-attachment-strip">
          {attachments.map((item) => (
            <div
              key={item.localId}
              className={
                item.kind === 'image'
                  ? 'composer-attachment-card image'
                  : 'composer-attachment-card file'
              }
            >
              {item.kind === 'image' ? (
                item.localPreviewUrl ? (
                  <img
                    src={item.localPreviewUrl}
                    alt={item.filename}
                    className="composer-attachment-thumb"
                  />
                ) : item.document_id > 0 ? (
                  <AuthenticatedImage
                    documentId={item.document_id}
                    alt={item.filename}
                    className="composer-attachment-thumb"
                    fallbackText={item.filename}
                  />
                ) : (
                  <div className="composer-attachment-file-icon">🖼</div>
                )
              ) : (
                <div className="composer-attachment-file-icon">📄</div>
              )}

              <div className="composer-attachment-info">
                <div className="composer-attachment-name">{item.filename}</div>
                <div className="composer-attachment-sub">
                  {item.uploadStatus === 'uploading'
                    ? `${item.uploadProgress}%`
                    : item.uploadStatus === 'failed'
                      ? item.error || '上传失败'
                      : '已上传'}
                </div>

                {item.uploadStatus === 'uploading' || item.uploadStatus === 'uploaded' ? (
                  <div className="composer-attachment-progress">
                    <div
                      className="composer-attachment-progress-bar"
                      style={{ width: `${item.uploadProgress}%` }}
                    />
                  </div>
                ) : item.uploadStatus === 'failed' ? (
                  <div className="composer-attachment-error">{item.error || '上传失败'}</div>
                ) : null}
              </div>

              <button
                type="button"
                className="composer-attachment-remove"
                onClick={() => removeAttachment(item.localId)}
                aria-label="Remove attachment"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}

      {dragActive ? <div className="composer-drop-hint">松开以上传文件</div> : null}

      <textarea
        value={userInput}
        onChange={(event) => setUserInput(event.target.value)}
        onPaste={handlePaste}
        onCompositionStart={() => setIsComposing(true)}
        onCompositionEnd={() => setIsComposing(false)}
        onKeyDown={handleComposerKeyDown}
        placeholder={placeholder}
      />

      <div className="composer-footer">
        <div className="composer-tools">
          <button
            type="button"
            aria-label="Upload files"
            onClick={() => fileInputRef.current?.click()}
          >
            +
          </button>
          <span>{text(locale, zh.research, 'Research')}</span>
          <span>{text(locale, zh.artifact, 'Artifact')}</span>
          <span>{text(locale, zh.skillDraft, 'Skill')}</span>
          <ModelSelector value={selectedModelConfigId} onChange={setSelectedModelConfigId} disabled={running} />
        </div>
        <button
          className={canSend ? 'send-button active' : 'send-button'}
          type="submit"
          disabled={!canSend}
          aria-label={text(locale, zh.send, 'Send')}
        >
          ↑
        </button>
      </div>
    </form>
  )

  return (
    <div className={hasConversation ? 'codex-chat-page has-chat' : 'codex-chat-page initial'}>
      {!hasConversation ? (
        <div className="initial-composer-stage">
          <h1>{initialTitle}</h1>
          {composer}
        </div>
      ) : (
        <>
          <div className="chat-workspace">
            <div className="chat-scroll">
              <div className="message-list">
                {replayLoading ? (
                  <div className="replay-notice" role="status">
                    {text(locale, '正在从事件账本恢复运行记录…', 'Restoring run history from the event ledger…')}
                  </div>
                ) : null}
                {replayError ? <div className="replay-notice error" role="alert">{replayError}</div> : null}
                {messages.map((message) => (
                  <AgentMessageItem
                    message={message}
                    locale={locale}
                    debug={debug}
                    key={messageKey(message)}
                    onApprove={(id) => handleApproval(id, true)}
                    onReject={(id) => handleApproval(id, false)}
                  />
                ))}
                {error ? (
                  <article className="chat-message assistant">
                    <div className="inline-error">{error}</div>
                  </article>
                ) : null}
                <div ref={bottomRef} />
              </div>
            </div>
            <div className="composer-dock">{composer}</div>
          </div>
          {debug ? (
            <div className="panel agent-debug-panel">
              <div className="row">
                <StatusPill value={currentRun?.status || (running ? 'running' : 'idle')} />
                {currentRun?.route ? <StatusPill value={currentRun.route} /> : null}
              </div>
              <details>
                <summary>Runtime debug</summary>
                <JsonBlock value={{ currentRun, activeConversationId, messages }} />
              </details>
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}
