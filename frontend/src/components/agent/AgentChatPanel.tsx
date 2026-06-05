import { FormEvent, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import * as agent from '../../api/agent'
import type { AgentEvent, AgentRun, AgentStep, Artifact, SkillDraft } from '../../api/types'
import { JsonBlock } from '../common/JsonBlock'
import { StatusPill } from '../common/StatusPill'

type ChatMessage = { id: string; role: 'user' | 'assistant'; content: string }

type AgentChatPanelProps = {
  source?: string
  pageContext?: Record<string, unknown>
  placeholder?: string
  initialTitle?: string
  debug?: boolean
  locale?: 'en' | 'zh'
}

function nodeLabel(name = '', locale: 'en' | 'zh') {
  const node = name.toLowerCase()
  const zh: Record<string, string> = {
    permission: '风险识别',
    home_intent: '需求判断',
    planner: '计划生成',
    context: '构建上下文',
    skill_matcher: '匹配 Skill',
    research: '深度研究',
    rag: '知识库检索',
    artifact: '生成成果',
    tool: '工具执行',
    memory: '写入记忆',
    skill: '沉淀 Skill',
    evaluator: '结果评估',
    final: '最终回复',
  }
  const en: Record<string, string> = {
    permission: 'Risk check',
    home_intent: 'Intent triage',
    planner: 'Plan',
    context: 'Build context',
    skill_matcher: 'Match Skill',
    research: 'Research',
    rag: 'Knowledge search',
    artifact: 'Create artifact',
    tool: 'Tool action',
    memory: 'Write memory',
    skill: 'Skill handling',
    evaluator: 'Evaluate',
    final: 'Final response',
  }
  const dict = locale === 'zh' ? zh : en
  for (const key of Object.keys(dict)) {
    if (node.includes(key)) return dict[key]
  }
  return locale === 'zh' ? '执行步骤' : 'Run step'
}

function eventLabel(eventType = '', locale: 'en' | 'zh') {
  if (locale !== 'zh') return eventType || 'completed'
  const labels: Record<string, string> = {
    run_started: '任务已开始',
    run_completed: '任务已完成',
    run_failed: '任务失败',
    run_cancelled: '任务已取消',
    node_started: '开始执行',
    node_completed: '执行完成',
    node_failed: '执行失败',
    approval_required: '等待审批',
    approval_approved: '审批已通过',
    approval_rejected: '审批已拒绝',
  }
  return labels[eventType] || eventType || '已完成'
}

function skillName(skill: unknown) {
  const item = skill as { name?: string; title?: string } | null
  return item?.name || item?.title || ''
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

function runAnswer(run: AgentRun) {
  const finalPayload = asRecord(run.final_payload)
  return String(finalPayload.answer || run.final_answer || run.final_output || '')
}

function runLangGraphStatus(run: AgentRun | null) {
  if (!run) return null
  const direct = asRecord(run.langgraphstatus)
  if (Object.keys(direct).length) return direct
  return asRecord(asRecord(run.final_payload).langgraphstatus)
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

type ExecutionTimelineProps = {
  events: AgentEvent[]
  steps: AgentStep[]
  running: boolean
  locale?: 'en' | 'zh'
  onApprove?: (approvalId: number) => void
  onReject?: (approvalId: number) => void
}

export function ExecutionTimeline({ events, steps, running, locale = 'en', onApprove, onReject }: ExecutionTimelineProps) {
  const timeline = events.length
    ? events.filter((event) => event.event_type !== 'node_started' || event.node_name)
    : steps.map((step) => ({ id: step.id, event_type: step.status || 'step', node_name: step.node_name, payload: { status: step.status } }) as AgentEvent)

  if (!running && !timeline.length) return null

  return (
    <article className="chat-message assistant">
      <div className="agent-trace">
        <div className="trace-header">
          <span className={running ? 'thinking-dot active' : 'thinking-dot'} />
          <strong>{running ? (locale === 'zh' ? 'Agent 正在执行' : 'Running Agent') : locale === 'zh' ? '执行时间线' : 'Execution timeline'}</strong>
        </div>
        <div className="trace-list">
          {timeline.length ? (
            timeline.map((event, index) => {
              const payload = event.payload || {}
              const approvalId = Number(payload.approval_id || (payload.approval_payload as { approval_id?: number } | undefined)?.approval_id || 0)
              const isApproval = event.event_type === 'approval_required'
              return (
                <div className={isApproval ? 'trace-item approval' : 'trace-item'} key={`${event.event_type}-${event.node_name}-${event.id || index}`}>
                  <span className="trace-icon">{event.event_type?.includes('failed') ? '!' : '✓'}</span>
                  <div>
                    <strong>{isApproval ? (locale === 'zh' ? '需要审批' : 'Approval required') : nodeLabel(event.node_name || event.event_type, locale)}</strong>
                    <p>{eventLabel(event.event_type || String(payload.status || 'completed'), locale)}</p>
                    {isApproval ? (
                      <div className="approval-inline-actions">
                        <small>{String(payload.risk_level || payload.permission_level || 'L3')}</small>
                        {approvalId ? (
                          <>
                            <button className="light-mini-button" onClick={() => onApprove?.(approvalId)}>
                              {locale === 'zh' ? '批准执行' : 'Approve'}
                            </button>
                            <button className="light-mini-button" onClick={() => onReject?.(approvalId)}>
                              {locale === 'zh' ? '拒绝' : 'Reject'}
                            </button>
                          </>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </div>
              )
            })
          ) : (
            <div className="trace-item">
              <span className="trace-icon loading">…</span>
              <div>
                <strong>{locale === 'zh' ? '正在准备任务' : 'Preparing task'}</strong>
                <p>{locale === 'zh' ? '正在加载运行上下文' : 'Loading runtime context'}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </article>
  )
}

function LangGraphStatusPanel({ run, locale = 'en' }: { run: AgentRun | null; locale?: 'en' | 'zh' }) {
  const status = runLangGraphStatus(run)
  const steps = Array.isArray(status?.steps) ? (status.steps as Record<string, unknown>[]) : []
  const riskLevel = status?.risk_level ? String(status.risk_level) : ''
  if (!steps.length) return null
  return (
    <article className="chat-message assistant">
      <div className="agent-trace langgraph-status-panel">
        <div className="trace-header">
          <span className="thinking-dot" />
          <strong>{locale === 'zh' ? '关键步骤' : 'Key steps'}</strong>
          {riskLevel ? <small>{locale === 'zh' ? `风险 ${riskLevel}` : `Risk ${riskLevel}`}</small> : null}
        </div>
        <div className="trace-list">
          {steps.map((step, index) => (
            <div className="trace-item" key={`${String(step.key || step.node_name || 'step')}-${index}`}>
              <span className="trace-icon">{step.status === 'failed' ? '!' : '✓'}</span>
              <div>
                <strong>{String(step.title || nodeLabel(String(step.node_name || ''), locale))}</strong>
                <p>{String(step.detail || step.status || '')}</p>
                {step.model || step.fallback_used ? (
                  <small className="trace-model-line">
                    {step.model ? String(step.model) : ''}
                    {step.fallback_used ? (locale === 'zh' ? ' · 已使用规则兜底' : ' · fallback used') : ''}
                  </small>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      </div>
    </article>
  )
}

function SkillMatchBadge({ run, locale = 'en' }: { run: AgentRun | null; locale?: 'en' | 'zh' }) {
  const matched = run?.matched_skill
  const candidates = run?.candidate_skills || []
  if (!matched && !candidates.length) return null

  return (
    <article className="chat-message assistant">
      <div className="agent-meta-card">
        {matched ? (
          <>
            <strong>{locale === 'zh' ? `已使用 Skill：${skillName(matched)}` : `Used Skill: ${skillName(matched)}`}</strong>
            <p>{String(matched.match_reason || (locale === 'zh' ? '根据可复用工作流信号命中。' : 'Matched by reusable workflow signals.'))}</p>
          </>
        ) : (
          <>
            <strong>{locale === 'zh' ? '可能适合使用的 Skill' : 'Possible Skills'}</strong>
            <p>{candidates.map((item) => skillName(item)).filter(Boolean).join(', ')}</p>
          </>
        )}
      </div>
    </article>
  )
}

function SkillDraftNotice({ draft, locale = 'en' }: { draft?: SkillDraft | null; locale?: 'en' | 'zh' }) {
  if (!draft) return null

  return (
    <article className="chat-message assistant">
      <div className="agent-meta-card">
        <strong>{locale === 'zh' ? `已生成 Skill 草稿：${draft.name || draft.description || `#${draft.id}`}` : `Skill draft created: ${draft.name || draft.description || `#${draft.id}`}`}</strong>
        <Link className="light-mini-button" to="/skills">
          {locale === 'zh' ? '去 Skills 审核' : 'Review in Skills'}
        </Link>
      </div>
    </article>
  )
}

function ArtifactInlineList({ artifacts, locale = 'en' }: { artifacts?: Artifact[]; locale?: 'en' | 'zh' }) {
  if (!artifacts?.length) return null

  return (
    <article className="chat-message assistant">
      <div className="agent-meta-card">
        <strong>{locale === 'zh' ? '生成的 Artifact' : 'Artifacts'}</strong>
        {artifacts.map((artifact) => (
          <Link className="light-mini-button" to="/artifacts" key={artifact.id}>
            {artifact.title || artifact.artifact_type || (locale === 'zh' ? `成果 #${artifact.id}` : `Artifact #${artifact.id}`)}
          </Link>
        ))}
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
  const [userInput, setUserInput] = useState('')
  const [run, setRun] = useState<AgentRun | null>(null)
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [stream, setStream] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])

  const hasConversation = messages.length > 0 || running || Boolean(run) || steps.length > 0 || events.length > 0 || Boolean(error)

  useEffect(() => {
    if (hasConversation) bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [hasConversation, messages, steps, events, running, run])

  async function handleApproval(approvalId: number, approved: boolean) {
    try {
      if (approved) await agent.approveRunApproval(approvalId, { decision: 'approved' })
      else await agent.rejectRunApproval(approvalId, { decision: 'rejected' })
      setEvents((items) => [...items, { event_type: approved ? 'approval_approved' : 'approval_rejected', payload: { approval_id: approvalId } }])
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : locale === 'zh' ? '审批操作失败。' : 'Approval action failed.')
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const input = userInput.trim()
    if (!input || running) return

    setUserInput('')
    setError('')
    setRunning(true)
    setRun(null)
    setSteps([])
    setEvents([])
    setStream([])
    setMessages((items) => [...items, { id: `${Date.now()}-user`, role: 'user', content: input }])

    try {
      const nextRun = await agent.createRun({
        user_input: input,
        input,
        source,
        page_context: pageContext,
        auto_skill: true,
        use_existing_skills: true,
        create_skill_draft_if_reusable: true,
      })
      setRun(nextRun)

      const runId = nextRun.run_id || nextRun.id
      if (runId) {
        const eventStream = agent.createRunEventStream(runId, {
          onMessage: async (message) => {
            setStream((items) => [...items.slice(-12), message.data])
            const parsed = parseEvent(message.data)
            if (parsed) setEvents((items) => [...items, parsed])
            try {
              const result = await agent.getSteps(runId)
              setSteps(result.steps || [])
            } catch {
              /* keep current timeline when step polling fails */
            }
          },
          onError: () => setStream((items) => [...items, 'events unavailable']),
        })
        setTimeout(async () => {
          eventStream.close()
          try {
            const result = await agent.getSteps(runId)
            setSteps(result.steps || [])
          } catch {
            /* ignore final polling failures */
          }
        }, 2600)
      }

      const fallback = locale === 'zh' ? 'Agent 已完成，但当前没有可展示的输出。' : 'Agent completed, but no displayable output was returned.'
      setMessages((items) => [...items, { id: `${Date.now()}-assistant`, role: 'assistant', content: runAnswer(nextRun) || fallback }])
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : locale === 'zh' ? 'Agent 运行失败，请稍后重试。' : 'Agent run failed. Please try again.')
    } finally {
      setRunning(false)
    }
  }

  const composer = (
    <form className={hasConversation ? 'codex-composer docked' : 'codex-composer centered'} onSubmit={submit}>
      <textarea value={userInput} onChange={(event) => setUserInput(event.target.value)} placeholder={placeholder} />
      <div className="composer-footer">
        <div className="composer-tools">
          <button type="button" aria-label={locale === 'zh' ? 'Agent 工具' : 'Agent tools'}>
            +
          </button>
          <span>{locale === 'zh' ? '研究' : 'Research'}</span>
          <span>{locale === 'zh' ? '成果' : 'Artifact'}</span>
          <span>{locale === 'zh' ? '技能' : 'Skill'}</span>
        </div>
        <button className={userInput.trim() ? 'send-button active' : 'send-button'} type="submit" disabled={!userInput.trim() || running} aria-label={locale === 'zh' ? '发送' : 'Send'}>
          →
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
          <div className="chat-scroll">
            <div className="message-list">
              {messages.map((message) => (
                <article className={message.role === 'user' ? 'chat-message user' : 'chat-message assistant'} key={message.id}>
                  <div className="message-bubble">{message.content}</div>
                </article>
              ))}
              <ExecutionTimeline events={events} steps={steps} running={running} locale={locale} onApprove={(id) => handleApproval(id, true)} onReject={(id) => handleApproval(id, false)} />
              <LangGraphStatusPanel run={run} locale={locale} />
              <SkillMatchBadge run={run} locale={locale} />
              <SkillDraftNotice draft={run?.created_skill_draft || null} locale={locale} />
              <ArtifactInlineList artifacts={run?.artifacts} locale={locale} />
              {error ? (
                <article className="chat-message assistant">
                  <div className="inline-error">{error}</div>
                </article>
              ) : null}
              <div ref={bottomRef} />
            </div>
          </div>
          <div className="composer-dock">{composer}</div>
          {debug ? (
            <div className="panel agent-debug-panel">
              <div className="row">
                <StatusPill value={run?.status || (running ? 'running' : 'idle')} />
                {run?.route ? <StatusPill value={run.route} /> : null}
              </div>
              <details>
                <summary>Runtime debug</summary>
                <JsonBlock value={{ run, stream, steps, events }} />
              </details>
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}
