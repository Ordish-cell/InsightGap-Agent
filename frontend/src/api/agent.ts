import { apiBaseUrl, apiRequest } from './client'
import { normalizeAgentStep } from './normalizers'
import type { AgentConversation, AgentReplayPage, AgentRun, AgentStep } from './types'

export const createRun = (payload: Record<string, unknown>) => apiRequest<AgentRun>('/agent/runs', { method: 'POST', body: payload })
export const getRun = (runId: number | string) => apiRequest<AgentRun>(`/agent/runs/${runId}`)
export const createConversation = (payload: Record<string, unknown> = {}) => apiRequest<AgentConversation>('/agent/conversations', { method: 'POST', body: payload })
export const listConversations = (query: { status?: string; limit?: number; offset?: number } = {}) =>
  apiRequest<{ items: AgentConversation[] }>('/agent/conversations', { query })
export const getConversation = (conversationId: string) => apiRequest<AgentConversation>(`/agent/conversations/${conversationId}`)
export const updateConversation = (conversationId: string, payload: Record<string, unknown>) =>
  apiRequest<AgentConversation>(`/agent/conversations/${conversationId}`, { method: 'PATCH', body: payload })
export const archiveConversation = (conversationId: string) => apiRequest<AgentConversation>(`/agent/conversations/${conversationId}/archive`, { method: 'POST' })
export const deleteConversation = (conversationId: string) => apiRequest<AgentConversation>(`/agent/conversations/${conversationId}`, { method: 'DELETE' })
export const clearConversation = (conversationId: string) => apiRequest<{ conversation: AgentConversation; cleared_messages: number }>(`/agent/conversations/${conversationId}/clear`, { method: 'POST' })
export const hardDeleteConversation = (conversationId: string) => apiRequest<{ conversation_id: string; deleted_records: number }>(`/agent/conversations/${conversationId}/hard`, { method: 'DELETE' })
export const hardDeleteConversationCancelPending = (conversationId: string) =>
  apiRequest<{ conversation_id: string; deleted_records: number; cancelled_approvals?: number; cancelled_runs?: number }>(
    `/agent/conversations/${conversationId}/hard?cancel_pending=true`,
    { method: 'DELETE' }
  )
export const approveRunApproval = (approvalId: number | string, payload: Record<string, unknown> = {}) => apiRequest<unknown>(`/agent/approvals/${approvalId}/approve`, { method: 'POST', body: payload })
export const rejectRunApproval = (approvalId: number | string, payload: Record<string, unknown> = {}) => apiRequest<unknown>(`/agent/approvals/${approvalId}/reject`, { method: 'POST', body: payload })
export const getSteps = async (runId: number | string) => {
  const result = await apiRequest<{ run_id: number; steps: AgentStep[] }>(`/agent/runs/${runId}/steps`)
  return { ...result, steps: (result.steps || []).map(normalizeAgentStep) }
}

export const getRunReplay = (
  runId: number | string,
  query: { after_seq?: number; until_seq?: number; limit?: number; event_type?: string } = {},
) => apiRequest<AgentReplayPage>(`/agent/runs/${runId}/events`, { query })

type LedgerHandlers = {
  onStarted?: (run: AgentRun & { last_event_seq?: number }) => void
  onMessage?: (event: MessageEvent) => void
  onError?: () => void
  onNetworkStatus?: (status: 'recovering' | 'caught_up' | 'retrying') => void
}

export class AgentLedgerClient {
  private lastSeq = new Map<number, number>()

  startAndTail(payload: Record<string, unknown>, handlers: LedgerHandlers) {
    const lifecycle = new AbortController()
    void (async () => {
      try {
        const started = await apiRequest<AgentRun & { last_event_seq?: number }>('/agent/runs/start', { method: 'POST', body: payload })
        const runId = Number(started.run_id || started.id || 0)
        if (!runId || lifecycle.signal.aborted) return
        const cursor = Number(started.last_event_seq || 0)
        this.lastSeq.set(runId, cursor)
        handlers.onStarted?.(started)
        handlers.onMessage?.({
          data: JSON.stringify({
            id: cursor,
            event_seq: cursor,
            schema_version: 1,
            run_id: runId,
            thread_id: started.thread_id,
            event_type: 'run_created',
            visibility: 'user',
            display_channel: 'status',
            payload: started,
          }),
        } as MessageEvent)
        await this.tail(runId, cursor, handlers, lifecycle.signal)
      } catch {
        if (!lifecycle.signal.aborted) handlers.onError?.()
      }
    })()
    return { close: () => lifecycle.abort() }
  }

  resumeAndTail(runId: number, handlers: LedgerHandlers) {
    const lifecycle = new AbortController()
    void (async () => {
      try {
        const knownCursor = this.lastSeq.get(runId)
        const resumed = await apiRequest<{ run_id: number; last_event_seq: number }>(`/agent/runs/${runId}/resume`, { method: 'POST' })
        const cursor = knownCursor ?? Number(resumed.last_event_seq || 0)
        this.lastSeq.set(runId, cursor)
        await this.tail(runId, cursor, handlers, lifecycle.signal)
      } catch {
        if (!lifecycle.signal.aborted) handlers.onError?.()
      }
    })()
    return { close: () => lifecycle.abort() }
  }

  async replay(runId: number, eventType?: string) {
    const events: AgentReplayPage['events'] = []
    let afterSeq = 0
    let untilSeq: number | undefined
    while (true) {
      const page = await getRunReplay(runId, { after_seq: afterSeq, until_seq: untilSeq, limit: 200, event_type: eventType })
      untilSeq = page.until_seq
      for (const event of page.events || []) {
        const seq = Number(event.event_seq ?? event.id ?? 0)
        if (seq > afterSeq) events.push(event)
      }
      afterSeq = page.next_seq
      if (!page.has_more) break
    }
    this.lastSeq.set(runId, Math.max(this.lastSeq.get(runId) || 0, afterSeq))
    return events
  }

  tailRun(runId: number, afterSeq: number, handlers: LedgerHandlers) {
    const lifecycle = new AbortController()
    void this.tail(runId, afterSeq, handlers, lifecycle.signal)
    return { close: () => lifecycle.abort() }
  }

  private async tail(runId: number, afterSeq: number, handlers: LedgerHandlers, signal: AbortSignal) {
    let cursor = Math.max(afterSeq, this.lastSeq.get(runId) || 0)
    const delays = [500, 1000, 2000, 4000, 8000]
    let attempts = 0
    while (!signal.aborted) {
      try {
        if (attempts > 0) handlers.onNetworkStatus?.('recovering')
        const terminal = await this.readStream(runId, cursor, signal, async (envelope) => {
          const seq = Number(envelope.event_seq ?? envelope.id ?? 0)
          if (!seq || seq <= cursor) return false
          cursor = seq
          this.lastSeq.set(runId, seq)
          if (['run_completed', 'run_failed', 'run_interrupted'].includes(String(envelope.event_type))) {
            try {
              const response = await getRun(runId)
              envelope.payload = { ...(envelope.payload || {}), response }
            } catch {
              // The terminal ledger event remains authoritative if canonical refresh fails.
            }
          }
          handlers.onMessage?.({ data: JSON.stringify(envelope) } as MessageEvent)
          return ['run_completed', 'run_failed', 'run_interrupted', 'run_paused'].includes(String(envelope.event_type))
        })
        handlers.onNetworkStatus?.('caught_up')
        if (terminal || signal.aborted) return
        throw new Error('Ledger stream ended before a terminal event')
      } catch {
        if (signal.aborted) return
        handlers.onNetworkStatus?.('retrying')
        const delay = delays[Math.min(attempts, delays.length - 1)]
        attempts += 1
        await new Promise<void>((resolve) => {
          const timer = window.setTimeout(resolve, delay)
          signal.addEventListener('abort', () => { window.clearTimeout(timer); resolve() }, { once: true })
        })
      }
    }
  }

  private async readStream(
    runId: number,
    cursor: number,
    signal: AbortSignal,
    onEnvelope: (event: Record<string, unknown>) => Promise<boolean>,
  ) {
    const token = localStorage.getItem('authToken')
    const headers = new Headers(token ? { Authorization: `Bearer ${token}` } : {})
    headers.set('Last-Event-ID', String(cursor))
    const response = await fetch(`${apiBaseUrl()}/agent/runs/${runId}/events/stream?after_seq=${cursor}`, { headers, signal })
    if (!response.ok || !response.body) throw new Error(`Stream failed: ${response.status}`)
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (!signal.aborted) {
      const { value, done } = await reader.read()
      if (done) return false
      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() || ''
      for (const chunk of chunks) {
        const data = chunk.split('\n').find((line) => line.startsWith('data: '))?.slice(6)
        if (!data) continue
        const envelope = JSON.parse(data) as Record<string, unknown>
        if (await onEnvelope(envelope)) return true
      }
    }
    return false
  }
}

export const agentLedgerClient = new AgentLedgerClient()

export function extractRunAnswer(response: AgentRun) {
  const finalResponse = (response.final_response || {}) as Record<string, unknown>
  const finalPayload = (response.final_payload || {}) as Record<string, unknown>
  const result = (response.result || {}) as Record<string, unknown>
  return String(
    response.answer ||
      finalResponse.answer ||
      response.assistant_message?.content ||
      result.answer ||
      finalPayload.answer ||
      response.final_answer ||
      response.final_output ||
      ''
  )
}
