import { FormEvent, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import * as agent from '../../api/agent'
import type { AgentRun, AgentStep, Artifact, SkillDraft } from '../../api/types'
import { statusLabel } from '../../utils/labels'
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

function stepLabel(step: AgentStep, locale: 'en' | 'zh') {
  const name = String(step.node_name || '').toLowerCase()
  if (name.includes('permission')) return locale === 'zh' ? '检查安全权限' : 'Permission check'
  if (name.includes('router')) return locale === 'zh' ? '识别任务路线' : 'Route task'
  if (name.includes('context') || name.includes('feed_card')) return locale === 'zh' ? '构建上下文' : 'Build context'
  if (name.includes('skill_matcher')) return locale === 'zh' ? '匹配 Skill' : 'Match Skill'
  if (name.includes('rag')) return locale === 'zh' ? '检索知识库' : 'Search knowledge'
  if (name.includes('research')) return locale === 'zh' ? '执行研究' : 'Run research'
  if (name.includes('tool')) return locale === 'zh' ? '调用工具' : 'Call tool'
  if (name.includes('artifact')) return locale === 'zh' ? '生成 Artifact' : 'Create artifact'
  if (name.includes('memory')) return locale === 'zh' ? '写入记忆' : 'Write memory'
  if (name.includes('skill')) return locale === 'zh' ? '处理 Skill' : 'Handle Skill'
  if (name.includes('eval')) return locale === 'zh' ? '评估结果' : 'Evaluate result'
  return locale === 'zh' ? '执行步骤' : 'Run step'
}

function skillName(skill: unknown) {
  const item = skill as { name?: string; title?: string } | null
  return item?.name || item?.title || ''
}

export function AgentRunTimeline({ steps, running, locale = 'en' }: { steps: AgentStep[]; running: boolean; locale?: 'en' | 'zh' }) {
  if (!running && !steps.length) return null
  return (
    <article className="chat-message assistant">
      <div className="agent-trace">
        <div className="trace-header">
          <span className={running ? 'thinking-dot active' : 'thinking-dot'} />
          <strong>{running ? (locale === 'zh' ? 'Agent 正在执行' : 'Running Agent') : locale === 'zh' ? 'Agent 已完成' : 'Agent completed'}</strong>
        </div>
        <div className="trace-list">
          {steps.length ? (
            steps.map((step) => (
              <div className="trace-item" key={step.id}>
                <span className="trace-icon">✓</span>
                <div>
                  <strong>{stepLabel(step, locale)}</strong>
                  <p>{statusLabel(step.status)}</p>
                </div>
              </div>
            ))
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
        <strong>{locale === 'zh' ? `已生成 Skill 草案：${draft.name || draft.description || `#${draft.id}`}` : `Skill draft created: ${draft.name || draft.description || `#${draft.id}`}`}</strong>
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

export function AgentChatPanel({ source = 'agent_page', pageContext = {}, placeholder = 'Enter a task for the Agent', initialTitle = 'What should we build or research?', debug = false, locale = 'en' }: AgentChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const [userInput, setUserInput] = useState('')
  const [run, setRun] = useState<AgentRun | null>(null)
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [stream, setStream] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])

  const hasConversation = messages.length > 0 || running || Boolean(run) || steps.length > 0 || Boolean(error)

  useEffect(() => {
    if (hasConversation) bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [hasConversation, messages, steps, running, run])

  async function submit(event: FormEvent) {
    event.preventDefault()
    const input = userInput.trim()
    if (!input || running) return
    setUserInput('')
    setError('')
    setRunning(true)
    setRun(null)
    setSteps([])
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
        const sourceStream = agent.createRunStream(runId, {
          onMessage: async (message) => {
            setStream((items) => [...items.slice(-12), message.data])
            try {
              const result = await agent.getSteps(runId)
              setSteps(result.steps || [])
            } catch {
              /* keep current trace when step polling fails */
            }
          },
          onError: () => setStream((items) => [...items, 'stream unavailable']),
        })
        setTimeout(async () => {
          sourceStream.close()
          try {
            const result = await agent.getSteps(runId)
            setSteps(result.steps || [])
          } catch {
            /* ignore final polling failures */
          }
        }, 2600)
      }
      setMessages((items) => [...items, { id: `${Date.now()}-assistant`, role: 'assistant', content: nextRun.final_output || (locale === 'zh' ? 'Agent 已完成，但当前没有可展示的输出。' : 'Agent completed, but no displayable output was returned.') }])
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
          <button type="button" aria-label="Agent tools">
            +
          </button>
          <span>{locale === 'zh' ? '研究' : 'Research'}</span>
          <span>{locale === 'zh' ? '成果' : 'Artifact'}</span>
          <span>{locale === 'zh' ? '技能' : 'Skill'}</span>
        </div>
        <button className={userInput.trim() ? 'send-button active' : 'send-button'} type="submit" disabled={!userInput.trim() || running} aria-label="Send">
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
              <AgentRunTimeline steps={steps} running={running} locale={locale} />
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
                <JsonBlock value={{ run, stream, steps }} />
              </details>
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}
