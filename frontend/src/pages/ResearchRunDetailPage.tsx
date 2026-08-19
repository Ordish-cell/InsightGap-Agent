import { useEffect, useRef, useState } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'

import * as research from '../api/research'
import type { ResearchRun, ResearchRunMetadata } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { EvidenceList } from '../components/common/EvidenceList'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { MarkdownRenderer } from '../components/common/MarkdownRenderer'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

const MAX_POLL_ATTEMPTS = 300
const MAX_CONSECUTIVE_FAILURES = 5
const POLL_INTERVAL_MS = 2000
const FAILURE_RETRY_INTERVAL_MS = 3000

function engineLabel(meta: ResearchRunMetadata | undefined, status: string | undefined): { text: string; className: string; description: string } {
  if (status === 'running') {
    return { text: '研究引擎准备中', className: 'engine-pending', description: '后台正在启动研究流程...' }
  }
  if (meta?.source === 'open_deep_research' || meta?.engine === 'open_deep_research') {
    return { text: 'Open Deep Research', className: 'engine-odr', description: '由 Open Deep Research 多智能体研究流程生成' }
  }
  if (meta?.used_fallback) {
    return { text: 'Fallback Researcher', className: 'engine-fallback', description: '当前结果为降级研究报告，Open Deep Research 未成功运行。' }
  }
  if (status === 'failed') {
    return { text: '研究失败', className: 'engine-failed', description: '研究流程未能完成。' }
  }
  return { text: '研究引擎', className: 'engine-pending', description: '' }
}

export function ResearchRunDetailPage() {
  const { researchRunId = '' } = useParams()
  const navigate = useNavigate()
  const [run, setRun] = useState<ResearchRun | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [pollingError, setPollingError] = useState('')
  const [retrying, setRetrying] = useState(false)

  // Refs to prevent double-polling from StrictMode / effect re-runs
  const timerRef = useRef<number | null>(null)
  const pollingRef = useRef(false)
  const attemptsRef = useRef(0)
  const failuresRef = useRef(0)

  useEffect(() => {
    if (!researchRunId) return
    if (pollingRef.current) return

    let cancelled = false
    pollingRef.current = true
    attemptsRef.current = 0
    failuresRef.current = 0

    setLoading(true)
    setError('')
    setPollingError('')

    const clearTimer = () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }

    const poll = async () => {
      if (cancelled) return
      attemptsRef.current += 1

      try {
        const data = await research.getRun(researchRunId)
        if (cancelled) return

        setRun(data)
        setLoading(false)
        failuresRef.current = 0

        if (data.status === 'completed' || data.status === 'failed') {
          pollingRef.current = false
          clearTimer()
          return
        }

        if (attemptsRef.current >= MAX_POLL_ATTEMPTS) {
          pollingRef.current = false
          clearTimer()
          setPollingError('研究仍在运行中，请稍后刷新页面查看结果。')
          return
        }

        timerRef.current = window.setTimeout(poll, POLL_INTERVAL_MS)
      } catch {
        if (cancelled) return
        failuresRef.current += 1

        if (failuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
          pollingRef.current = false
          clearTimer()
          setPollingError('连续多次无法获取研究状态，请检查网络后刷新页面。')
          setLoading(false)
          return
        }

        timerRef.current = window.setTimeout(poll, FAILURE_RETRY_INTERVAL_MS)
      }
    }

    poll()

    return () => {
      cancelled = true
      pollingRef.current = false
      clearTimer()
    }
  }, [researchRunId])

  async function handleRetry() {
    if (!run) return
    setRetrying(true)
    try {
      const payload: Record<string, unknown> = {
        query: run.query,
        source: run.metadata?.source || 'manual',
        feed_card_id: run.metadata?.feed_card_id,
        card_snapshot: run.metadata?.card_snapshot,
        auto_start: true,
        force_engine: 'open_deep_research',
      }
      const newRun = await research.createRun(payload)
      if (newRun?.id) navigate(`/research/${newRun.id}`)
    } catch {
      setRetrying(false)
    }
  }

  if (loading) return <LoadingState title="正在加载研究详情" />
  if (error) return <ErrorState message={error} />
  if (!run) return <EmptyState title="未找到研究记录" />

  const meta = run.metadata as ResearchRunMetadata | undefined
  const engine = engineLabel(meta, run.status)
  const isRunning = run.status === 'running'
  const isFailed = run.status === 'failed'
  const isFallback = meta?.used_fallback === true
  const isOdr = meta?.source === 'open_deep_research' || meta?.engine === 'open_deep_research'

  return (
    <section className="workbench-page research-detail-page">
      <PageHeader
        title={run.query || '深度研究'}
        description={
          <span className="inline-meta">
            <span className={`engine-badge ${engine.className}`} title={engine.description}>
              {engine.text}
            </span>
            <StatusPill value={run.status || 'pending'} />
            {isOdr && <span className="badge odr-verified">真实 ODR</span>}
            {isFallback && <span className="badge fallback-warn">降级报告</span>}
          </span>
        }
        actions={
          <>
            {(isFailed || isFallback) && (
              <button className="button" onClick={handleRetry} disabled={retrying}>
                {retrying ? '重试中...' : '重新使用 Open Deep Research 研究'}
              </button>
            )}
          </>
        }
      />

      {isRunning && (
        <div className="panel running-panel">
          <div className="running-indicator">
            <span className="spinner" />
            <span>正在执行深度研究...</span>
          </div>
          <div className="running-steps">
            <p>研究引擎正在执行以下步骤：</p>
            <ul>
              <li className={meta?.source ? 'done' : ''}>初始化研究引擎</li>
              <li>拆解研究问题</li>
              <li>多源并行检索资料</li>
              <li>整合证据并生成报告</li>
            </ul>
          </div>
          {pollingError && (
            <p className="muted small" style={{ marginTop: 12 }}>{pollingError}</p>
          )}
        </div>
      )}

      {isFailed && (
        <div className="panel error-panel">
          <h2>研究失败</h2>
          <p>{run.error_message || run.error || '研究流程未能完成，请重试。'}</p>
        </div>
      )}

      {isFallback && meta?.odr_error && (
        <div className="panel warning-panel">
          <h2>降级说明</h2>
          <p>Open Deep Research 未能成功运行，当前结果为降级研究报告。</p>
          {meta.odr_error && <p className="muted small">原因：{meta.odr_error}</p>}
        </div>
      )}

      <div className="split">
        <div className="stack">
          {run.summary && (
            <div className="panel">
              <h2>摘要</h2>
              <p>{run.summary}</p>
            </div>
          )}

          <div className="panel">
            <h2>研究报告</h2>
            {run.markdown_report ? (
              <MarkdownRenderer content={run.markdown_report} />
            ) : isRunning ? (
              <EmptyState title="报告生成中..." />
            ) : (
              <EmptyState title="暂无报告" />
            )}
          </div>

          {run.evidence && run.evidence.length > 0 && (
            <div className="panel">
              <h2>证据</h2>
              <EvidenceList evidence={run.evidence} />
            </div>
          )}

          {run.sources && run.sources.length > 0 && (
            <div className="panel">
              <h2>来源</h2>
              <ul>
                {run.sources.map((s: Record<string, unknown>, i: number) => (
                  <li key={i}>
                    {s.url ? (
                      <a href={String(s.url)} target="_blank" rel="noreferrer">
                        {String(s.title || s.url)}
                      </a>
                    ) : (
                      String(s.title || `来源 ${i + 1}`)
                    )}
                    {Boolean(s.note) && <span className="muted small"> — {String(s.note).slice(0, 200)}</span>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <aside className="stack">
          <div className="panel stack">
            <span className="mono small">{run.id}</span>

            {run.artifact_id ? (
              <Link className="button secondary" to={`/artifacts?artifactId=${run.artifact_id}`}>
                查看成果
              </Link>
            ) : (
              <span className="muted">暂无关联成果</span>
            )}
            {run.skill_draft_id ? (
              <Link className="button secondary" to="/skills">
                查看技能草稿
              </Link>
            ) : (
              <span className="muted">暂无技能草稿</span>
            )}
          </div>

          {meta && (
            <div className="panel">
              <h3>研究引擎信息</h3>
              <table className="kv-table">
                <tbody>
                  <tr><td>引擎</td><td>{meta.engine || meta.source || '未知'}</td></tr>
                  <tr><td>适配器</td><td>{meta.adapter || '未知'}</td></tr>
                  <tr><td>降级</td><td>{meta.used_fallback ? '是' : '否'}</td></tr>
                  {meta.odr_enabled !== undefined && (
                    <tr><td>ODR 启用</td><td>{meta.odr_enabled ? '是' : '否'}</td></tr>
                  )}
                  {meta.odr_error && (
                    <tr><td>ODR 错误</td><td className="muted small">{meta.odr_error}</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          )}

          <details className="panel">
            <summary>技术详情</summary>
            <JsonBlock
              value={{
                findings: run.findings,
                risks: run.risks,
                opportunities: run.opportunities,
                suggested_actions: run.suggested_actions,
                metadata: meta,
              }}
            />
          </details>
        </aside>
      </div>
    </section>
  )
}
