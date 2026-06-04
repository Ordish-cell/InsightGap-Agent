import { apiBaseUrl, apiRequest } from './client'
import { normalizeAgentStep } from './normalizers'
import type { AgentRun, AgentStep } from './types'

export const createRun = (payload: Record<string, unknown>) => apiRequest<AgentRun>('/agent/runs', { method: 'POST', body: payload })
export const createLegacyRun = (payload: Record<string, unknown>) => apiRequest<AgentRun>('/agent/run', { method: 'POST', body: payload })
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

function createFetchStream(path: string, handlers: { onMessage?: (event: MessageEvent) => void; onError?: () => void }) {
  const token = localStorage.getItem('authToken')
  const controller = new AbortController()
  fetch(`${apiBaseUrl()}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
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
