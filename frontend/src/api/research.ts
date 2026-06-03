import { apiRequest, normalizeList } from './client'
import type { ResearchRun } from './types'

export const createRun = (payload: Record<string, unknown>) => apiRequest<ResearchRun>('/research/runs', { method: 'POST', body: payload })
export const listRuns = async (params?: Record<string, unknown>) => normalizeList<ResearchRun>(await apiRequest<unknown>('/research/runs', { query: params as Record<string, string> }))
export const getRun = (researchRunId: string) => apiRequest<ResearchRun>(`/research/runs/${researchRunId}`)
