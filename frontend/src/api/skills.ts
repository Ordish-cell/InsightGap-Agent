import { apiRequest, normalizeList } from './client'
import type { SkillDraft } from './types'

export const list = async (params?: Record<string, unknown>) => normalizeList<SkillDraft>(await apiRequest<unknown>('/skills', { query: params as Record<string, string> }))
export const approve = (skillId: number | string) => apiRequest<SkillDraft>(`/skills/${skillId}/approve`, { method: 'POST' })
export const disable = (skillId: number | string) => apiRequest<SkillDraft>(`/skills/${skillId}/disable`, { method: 'POST' })
