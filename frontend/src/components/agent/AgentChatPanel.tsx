import { FormEvent, useEffect, useRef, useState } from 'react'

import * as agent from '../../api/agent'
import type { AgentChatMessage, AgentEvent, AgentRun, AgentRunStep, UnknownRecord } from '../../api/types'
import { JsonBlock } from '../common/JsonBlock'
import { StatusPill } from '../common/StatusPill'
import { AgentThoughtStream } from './AgentThoughtStream'

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
  creatingRun: '\u6b63\u5728\u521b\u5efa Agent Run\uff0c\u5e76\u63a5\u5165\u5f53\u524d\u4f1a\u8bdd\u4e0a\u4e0b\u6587\u3002',
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
  return message.message_id || message.local_id || `${message.role}-${message.id || message.created_at}`
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
  const direct = Array.isArray(message.steps) ? message.steps : []
  if (direct.length) return direct
  const status = asRecord(message.langgraphstatus)
  const fromStatus = Array.isArray(status.steps) ? (status.steps as AgentRunStep[]) : []
  if (fromStatus.length) return fromStatus
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
  if (message.role === 'user') {
    return (
      <article className="chat-message user">
        <div className="message-bubble">{message.content}</div>
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
          {message.content || (message.status === 'thinking' ? text(locale, zh.processing, 'Processing...') : text(locale, zh.noAnswer, 'No answer to display.'))}
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
  const [error, setError] = useState('')

  const hasConversation = messages.length > 0 || running || Boolean(activeConversationId) || Boolean(error)
  const selectedFeedTitle = String(pageContext.selected_feed_card_title || '')

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
    const item = await agent.getConversation(conversationId)
    setActiveConversationId(item.conversation_id)
    setMessages((item.messages || []) as UiMessage[])
  }

  function startNewConversation() {
    streamRef.current?.close()
    setActiveConversationId('')
    setMessages([])
    setCurrentRun(null)
    setError('')
  }

  async function handleApproval(approvalId: number, approved: boolean) {
    try {
      if (approved) await agent.approveRunApproval(approvalId, { decision: 'approved' })
      else await agent.rejectRunApproval(approvalId, { decision: 'rejected' })
      setMessages((items) =>
        items.map((message) =>
          message.role === 'assistant' && message.status === 'waiting_approval'
            ? {
                ...message,
                trace_events: [...(message.trace_events || []), { event_type: approved ? 'approval_approved' : 'approval_rejected', payload: { approval_id: approvalId } }],
              }
            : message
        )
      )
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : text(locale, zh.approvalFailed, 'Approval action failed.'))
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const input = userInput.trim()
    if (!input || running) return

    streamRef.current?.close()
    setUserInput('')
    setError('')
    setRunning(true)

    const now = Date.now()
    const localConversationId = activeConversationId
    const localUser: UiMessage = {
      message_id: `local-user-${now}`,
      role: 'user',
      content: input,
      conversation_id: localConversationId,
      status: 'completed',
      local_id: `local-user-${now}`,
    }
    const localAssistant: UiMessage = {
      message_id: `local-assistant-${now}`,
      role: 'assistant',
      content: '',
      conversation_id: localConversationId,
      status: 'thinking',
      trace_events: [
        {
          event_type: 'visible_thought',
          payload: {
            text: text(locale, zh.creatingRun, 'Creating the Agent Run and attaching conversation context.'),
            status: 'running',
          },
        },
      ],
      local_id: `local-assistant-${now}`,
    }
    setMessages((items) => [...items, localUser, localAssistant])

    let liveRunId: number | undefined
    let liveAssistantMessageId = localAssistant.message_id

    streamRef.current = agent.createRunLiveStream(
      {
        user_input: input,
        input,
        conversation_id: activeConversationId || undefined,
        source,
        page_context: pageContext,
        auto_skill: true,
        use_existing_skills: true,
        create_skill_draft_if_reusable: true,
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
                    status: 'thinking',
                    trace_events: [...(localAssistant.trace_events || []), parsed],
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
                if (item.message_id === assistantMessage.message_id || item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id) {
                  return { ...assistantMessage, trace_events: [...(item.trace_events || []), parsed].slice(-48) }
                }
                return item
              })
            )
            setRunning(false)
            streamRef.current?.close()
            streamRef.current = null
            return
          }

          if (parsed.event_type === 'run_failed') {
            const failedText = String(payload.error || payload.answer || text(locale, zh.agentFailed, 'Agent run failed. Please try again.'))
            setMessages((items) =>
              items.map((item) =>
                item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id || (liveRunId && item.run_id === liveRunId)
                  ? { ...item, status: 'failed', content: failedText, error_message: failedText, trace_events: [...(item.trace_events || []), parsed].slice(-48) }
                  : item
              )
            )
            setError(failedText)
            setRunning(false)
            return
          }

          setMessages((items) =>
            items.map((item) =>
              item.message_id === liveAssistantMessageId || item.message_id === localAssistant.message_id || (liveRunId && item.run_id === liveRunId)
                ? {
                    ...item,
                    run_id: liveRunId || item.run_id,
                    status: 'thinking',
                    trace_events: [...(item.trace_events || []), parsed].slice(-48),
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

  const composer = (
    <form className={hasConversation ? 'codex-composer docked' : 'codex-composer centered'} onSubmit={submit}>
      {selectedFeedTitle ? <div className="selected-context-pill">{text(locale, `${zh.using}${selectedFeedTitle}`, `Using: ${selectedFeedTitle}`)}</div> : null}
      <textarea value={userInput} onChange={(event) => setUserInput(event.target.value)} placeholder={placeholder} />
      <div className="composer-footer">
        <div className="composer-tools">
          <button type="button" aria-label={text(locale, zh.tools, 'Agent tools')}>
            +
          </button>
          <span>{text(locale, zh.research, 'Research')}</span>
          <span>{text(locale, zh.artifact, 'Artifact')}</span>
          <span>{text(locale, zh.skillDraft, 'Skill')}</span>
        </div>
        <button className={userInput.trim() ? 'send-button active' : 'send-button'} type="submit" disabled={!userInput.trim() || running} aria-label={text(locale, zh.send, 'Send')}>
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
