import { apiRequest, normalizeList } from './client'
import { normalizeResearchRun } from './normalizers'
import type { ResearchRun } from './types'

export const createRun = async (payload: Record<string, unknown>) => normalizeResearchRun(await apiRequest<ResearchRun>('/research/runs', { method: 'POST', body: payload }))
export const listRuns = async (params?: Record<string, unknown>) => normalizeList<ResearchRun>(await apiRequest<unknown>('/research/runs', { query: params as Record<string, string> })).map(normalizeResearchRun)
export const getRun = async (researchRunId: string) => normalizeResearchRun(await apiRequest<ResearchRun>(`/research/runs/${researchRunId}`))
