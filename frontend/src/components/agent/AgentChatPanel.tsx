import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'

import * as agent from '../../api/agent'
import type { AgentChatMessage, AgentConversation, AgentEvent, AgentRun, AgentRunStep, UnknownRecord } from '../../api/types'
import { JsonBlock } from '../common/JsonBlock'
import { StatusPill } from '../common/StatusPill'

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

function text(locale: 'en' | 'zh', zh: string, en: string) {
  return locale === 'zh' ? zh : en
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
  const zh: Record<string, string> = {
    run_started: '开始处理请求',
    home_intent_started: '理解请求',
    home_intent_completed: '完成意图识别',
    node_started: '执行步骤',
    node_completed: '完成步骤',
    node_failed: '步骤失败',
    approval_required: '等待审批',
    final_response_created: '生成最终回答',
    run_completed: '执行完成',
    run_failed: '执行失败',
    permission_guard: '检查风险',
    planner: '制定计划',
    context_builder: '检查上下文',
    skill_matcher: '匹配 Skill',
    research_agent: '执行研究',
    rag_agent: '检索知识库',
    artifact_agent: '生成产物',
    tool_agent: '准备工具动作',
    memory_agent: '写入记忆',
    skill_agent: '沉淀 Skill',
    evaluator: '验证结果',
    final_response: '整理回答',
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
  const dict = locale === 'zh' ? zh : en
  return dict[node] || dict[String(event.event_type || '')] || node || text(locale, '执行记录', 'Run event')
}

function stepSummary(step: AgentRunStep) {
  return String(step.summary || step.detail || step.status || '')
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
        summary: String(payload.summary || payload.answer || payload.final_output || payload.reason || ''),
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
    ? text(locale, '正在处理', 'Processing')
    : failed
      ? text(locale, '执行失败', 'Run failed')
      : compact
        ? text(locale, '已回答', 'Answered')
        : text(locale, '已完成思考', 'Completed reasoning')
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
          {elapsed ? ` · ${elapsed}` : ''}
        </strong>
        {!compact || failed ? <span className="run-trace-chevron">{open ? '⌄' : '›'}</span> : null}
      </button>
      {open && (!compact || failed) ? (
        <div className="run-step-list">
          {steps.length ? (
            steps.map((step, index) => (
              <div className={`run-step ${step.status || 'completed'}`} key={`${step.key || step.title || 'step'}-${index}`}>
                <span className="run-step-mark">{step.status === 'failed' ? '!' : step.status === 'running' ? '·' : '✓'}</span>
                <div>
                  <strong>{String(step.title || step.node_name || text(locale, '执行步骤', 'Run step'))}</strong>
                  {stepSummary(step) ? <p>{stepSummary(step)}</p> : null}
                </div>
              </div>
            ))
          ) : (
            <div className="run-step running">
              <span className="run-step-mark">·</span>
              <div>
                <strong>{text(locale, '正在等待状态记录', 'Waiting for status records')}</strong>
                <p>{text(locale, '运行开始后会持续追加可读步骤。', 'Readable steps appear as the run progresses.')}</p>
              </div>
            </div>
          )}
          {approvalId ? (
            <div className="approval-inline-actions">
              <small>{String(approvalPayload.risk_level || approvalPayload.permission_level || 'L3')}</small>
              <button className="light-mini-button" type="button" onClick={() => onApprove(approvalId)}>
                {text(locale, '批准执行', 'Approve')}
              </button>
              <button className="light-mini-button" type="button" onClick={() => onReject(approvalId)}>
                {text(locale, '拒绝', 'Reject')}
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
  onApprove,
  onReject,
}: {
  message: UiMessage
  locale: 'en' | 'zh'
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
        <AgentRunTraceBlock message={message} locale={locale} onApprove={onApprove} onReject={onReject} />
        <div className="message-bubble answer-content">
          {message.content || (message.status === 'thinking' ? text(locale, '正在处理...', 'Processing...') : text(locale, '没有可显示的回答。', 'No answer to display.'))}
        </div>
      </div>
    </article>
  )
}

function ConversationSidebar({
  conversations,
  activeConversationId,
  locale,
  loading,
  onNew,
  onRefresh,
  onSelect,
  onClear,
  onArchive,
}: {
  conversations: AgentConversation[]
  activeConversationId: string
  locale: 'en' | 'zh'
  loading: boolean
  onNew: () => void
  onRefresh: () => void
  onSelect: (conversationId: string) => void
  onClear: () => void
  onArchive: () => void
}) {
  return (
    <aside className="agent-conversation-sidebar">
      <div className="conversation-sidebar-head">
        <strong>{text(locale, '会话', 'Conversations')}</strong>
        <div className="conversation-sidebar-actions">
          <button type="button" title={text(locale, '刷新会话', 'Refresh conversations')} onClick={onRefresh}>
            ↻
          </button>
          <button type="button" title={text(locale, '新建会话', 'New conversation')} onClick={onNew}>
            +
          </button>
        </div>
      </div>
      <div className="conversation-list">
        {loading ? <span className="conversation-empty">{text(locale, '正在加载会话', 'Loading conversations')}</span> : null}
        {!loading && !conversations.length ? <span className="conversation-empty">{text(locale, '还没有会话', 'No conversations yet')}</span> : null}
        {conversations.map((item) => (
          <button
            className={item.conversation_id === activeConversationId ? 'conversation-item active' : 'conversation-item'}
            type="button"
            key={item.conversation_id}
            onClick={() => onSelect(item.conversation_id)}
          >
            <strong>{item.title || text(locale, '未命名会话', 'Untitled conversation')}</strong>
            <span>{item.last_message_preview || text(locale, '暂无消息', 'No messages yet')}</span>
          </button>
        ))}
      </div>
      <div className="conversation-sidebar-foot">
        <button type="button" onClick={onClear} disabled={!activeConversationId}>
          {text(locale, '清空消息', 'Clear')}
        </button>
        <button type="button" onClick={onArchive} disabled={!activeConversationId}>
          {text(locale, '归档', 'Archive')}
        </button>
      </div>
    </aside>
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
  const [conversations, setConversations] = useState<AgentConversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState('')
  const [currentRun, setCurrentRun] = useState<AgentRun | null>(null)
  const [running, setRunning] = useState(false)
  const [loadingConversations, setLoadingConversations] = useState(false)
  const [error, setError] = useState('')

  const hasConversation = messages.length > 0 || running || Boolean(activeConversationId) || Boolean(error)
  const selectedFeedTitle = String(pageContext.selected_feed_card_title || '')

  const sortedConversations = useMemo(() => conversations.filter((item) => item.status !== 'deleted'), [conversations])

  useEffect(() => {
    void refreshConversations(true)
    return () => streamRef.current?.close()
  }, [])

  useEffect(() => {
    if (hasConversation) bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [hasConversation, messages, running, error])

  async function refreshConversations(loadFirst = false) {
    setLoadingConversations(true)
    try {
      const result = await agent.listConversations({ status: 'active', limit: 50 })
      const items = result.items || []
      setConversations(items)
      if (loadFirst && !activeConversationId && items[0]?.conversation_id) {
        await loadConversation(items[0].conversation_id)
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : text(locale, '会话列表加载失败。', 'Conversation list failed.'))
    } finally {
      setLoadingConversations(false)
    }
  }

  async function loadConversation(conversationId: string) {
    streamRef.current?.close()
    const item = await agent.getConversation(conversationId)
    setActiveConversationId(item.conversation_id)
    setMessages((item.messages || []) as UiMessage[])
    setConversations((current) => {
      const exists = current.some((row) => row.conversation_id === item.conversation_id)
      return exists ? current.map((row) => (row.conversation_id === item.conversation_id ? item : row)) : [item, ...current]
    })
  }

  function startNewConversation() {
    streamRef.current?.close()
    setActiveConversationId('')
    setMessages([])
    setCurrentRun(null)
    setError('')
  }

  async function clearActiveConversation() {
    if (!activeConversationId) return
    await agent.clearConversation(activeConversationId)
    setMessages([])
    await refreshConversations()
  }

  async function archiveActiveConversation() {
    if (!activeConversationId) return
    await agent.archiveConversation(activeConversationId)
    startNewConversation()
    await refreshConversations()
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
      setError(exc instanceof Error ? exc.message : text(locale, '审批操作失败。', 'Approval action failed.'))
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

    const localConversationId = activeConversationId
    const localUser: UiMessage = {
      message_id: `local-user-${Date.now()}`,
      role: 'user',
      content: input,
      conversation_id: localConversationId,
      status: 'completed',
      local_id: `local-user-${Date.now()}`,
    }
    const localAssistant: UiMessage = {
      message_id: `local-assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      conversation_id: localConversationId,
      status: 'thinking',
      steps: [{ key: 'understanding', title: text(locale, '理解请求', 'Understanding request'), status: 'running', summary: text(locale, '正在创建 Agent Run 并连接会话上下文。', 'Creating the Agent Run and attaching conversation context.') }],
      local_id: `local-assistant-${Date.now()}`,
    }
    setMessages((items) => [...items, localUser, localAssistant])

    try {
      const response = await agent.createRun({
        user_input: input,
        input,
        conversation_id: activeConversationId || undefined,
        source,
        page_context: pageContext,
        auto_skill: true,
        use_existing_skills: true,
        create_skill_draft_if_reusable: true,
      })
      setCurrentRun(response)
      const { userMessage, assistantMessage } = responseMessages(response, localUser, localAssistant)
      const nextConversationId = response.conversation_id || response.conversation?.conversation_id || assistantMessage.conversation_id || ''
      if (nextConversationId) setActiveConversationId(nextConversationId)
      setMessages((items) =>
        items.map((message) => {
          if (message.message_id === localUser.message_id) return userMessage
          if (message.message_id === localAssistant.message_id) return assistantMessage
          return message
        })
      )
      if (response.conversation) {
        setConversations((items) => {
          const exists = items.some((item) => item.conversation_id === response.conversation?.conversation_id)
          return exists ? items.map((item) => (item.conversation_id === response.conversation?.conversation_id ? response.conversation as AgentConversation : item)) : [response.conversation as AgentConversation, ...items]
        })
      }

      const runId = response.run_id || response.id
      if (runId) {
        try {
          const result = await agent.getSteps(runId)
          setMessages((items) =>
            items.map((message) =>
              message.message_id === assistantMessage.message_id || message.run_id === runId
                ? { ...message, steps: result.steps as AgentRunStep[] }
                : message
            )
          )
        } catch {
          /* events still carry trace information */
        }
        streamRef.current = agent.createRunEventStream(runId, {
          onMessage: (message) => {
            const parsed = parseEvent(message.data)
            if (!parsed) return
            setMessages((items) =>
              items.map((item) =>
                item.message_id === assistantMessage.message_id || item.run_id === runId
                  ? {
                      ...item,
                      trace_events: [...(item.trace_events || []), parsed].slice(-24),
                      content: item.content || agent.extractRunAnswer(response),
                    }
                  : item
              )
            )
          },
          onError: () => undefined,
        })
        window.setTimeout(() => {
          streamRef.current?.close()
          streamRef.current = null
        }, 1800)
      }
      await refreshConversations()
    } catch (exc) {
      const failedText = exc instanceof Error ? exc.message : text(locale, 'Agent 运行失败，请稍后重试。', 'Agent run failed. Please try again.')
      setMessages((items) =>
        items.map((message) =>
          message.message_id === localAssistant.message_id
            ? { ...message, status: 'failed', content: failedText, error_message: failedText, steps: [{ key: 'failed', title: text(locale, '执行失败', 'Run failed'), status: 'failed', summary: failedText }] }
            : message
        )
      )
      setError(failedText)
    } finally {
      setRunning(false)
    }
  }

  const composer = (
    <form className={hasConversation ? 'codex-composer docked' : 'codex-composer centered'} onSubmit={submit}>
      {selectedFeedTitle ? <div className="selected-context-pill">{text(locale, `已带入信息：${selectedFeedTitle}`, `Using: ${selectedFeedTitle}`)}</div> : null}
      <textarea value={userInput} onChange={(event) => setUserInput(event.target.value)} placeholder={placeholder} />
      <div className="composer-footer">
        <div className="composer-tools">
          <button type="button" aria-label={text(locale, 'Agent 工具', 'Agent tools')}>
            +
          </button>
          <span>{text(locale, '研究', 'Research')}</span>
          <span>{text(locale, '成果', 'Artifact')}</span>
          <span>{text(locale, '技能', 'Skill')}</span>
        </div>
        <button className={userInput.trim() ? 'send-button active' : 'send-button'} type="submit" disabled={!userInput.trim() || running} aria-label={text(locale, '发送', 'Send')}>
          ↑
        </button>
      </div>
    </form>
  )

  return (
    <div className={hasConversation ? 'codex-chat-page has-chat with-conversations' : 'codex-chat-page initial'}>
      {!hasConversation ? (
        <div className="initial-composer-stage">
          <h1>{initialTitle}</h1>
          {composer}
        </div>
      ) : (
        <>
          <ConversationSidebar
            conversations={sortedConversations}
            activeConversationId={activeConversationId}
            locale={locale}
            loading={loadingConversations}
            onNew={startNewConversation}
            onRefresh={() => void refreshConversations()}
            onSelect={(conversationId) => void loadConversation(conversationId)}
            onClear={() => void clearActiveConversation()}
            onArchive={() => void archiveActiveConversation()}
          />
          <div className="chat-workspace">
            <div className="chat-scroll">
              <div className="message-list">
                {messages.map((message) => (
                  <AgentMessageItem
                    message={message}
                    locale={locale}
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
