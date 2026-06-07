import { apiBaseUrl, apiRequest } from './client'
import { normalizeAgentStep } from './normalizers'
import type { AgentConversation, AgentRun, AgentStep } from './types'

export const createRun = (payload: Record<string, unknown>) => apiRequest<AgentRun>('/agent/runs', { method: 'POST', body: payload })
export const createLegacyRun = (payload: Record<string, unknown>) => apiRequest<AgentRun>('/agent/run', { method: 'POST', body: payload })
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
export const approveRunApproval = (approvalId: number | string, payload: Record<string, unknown> = {}) => apiRequest<unknown>(`/agent/approvals/${approvalId}/approve`, { method: 'POST', body: payload })
export const rejectRunApproval = (approvalId: number | string, payload: Record<string, unknown> = {}) => apiRequest<unknown>(`/agent/approvals/${approvalId}/reject`, { method: 'POST', body: payload })
export const getSteps = async (runId: number | string) => {
  const result = await apiRequest<{ run_id: number; steps: AgentStep[] }>(`/agent/runs/${runId}/steps`)
  return { ...result, steps: (result.steps || []).map(normalizeAgentStep) }
}

export function createRunStream(runId: number | string, handlers: { onMessage?: (event: MessageEvent) => void; onError?: () => void }) {
  return createFetchStream(`/agent/runs/${runId}/stream`, handlers)
}

export function createRunEventStream(runId: number | string, handlers: { onMessage?: (event: MessageEvent) => void; onError?: () => void }) {
  return createFetchStream(`/agent/runs/${runId}/events`, handlers)
}

export function createRunLiveStream(payload: Record<string, unknown>, handlers: { onMessage?: (event: MessageEvent) => void; onError?: () => void }) {
  return createFetchStream('/agent/runs/stream', handlers, { method: 'POST', body: payload })
}

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

function createFetchStream(path: string, handlers: { onMessage?: (event: MessageEvent) => void; onError?: () => void }, options: { method?: string; body?: unknown } = {}) {
  const token = localStorage.getItem('authToken')
  const controller = new AbortController()
  const headers = new Headers(token ? { Authorization: `Bearer ${token}` } : {})
  let body: BodyInit | undefined
  if (options.body !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(options.body)
  }
  fetch(`${apiBaseUrl()}${path}`, {
    method: options.method || 'GET',
    headers,
    body,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok || !response.body) throw new Error(`Stream failed: ${response.status}`)
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() || ''
        chunks.forEach((chunk) => {
          const data = chunk.split('\n').find((line) => line.startsWith('data: '))?.slice(6) || chunk
          handlers.onMessage?.({ data } as MessageEvent)
        })
      }
    })
    .catch(() => {
      if (!controller.signal.aborted) handlers.onError?.()
    })
  return { close: () => controller.abort() }
}
