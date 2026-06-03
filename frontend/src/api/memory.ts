import { apiRequest } from './client'
import type { MemoryItem, MemorySummary } from './types'

export const summary = () => apiRequest<MemorySummary>('/memory/summary')
export const search = (payload: Record<string, unknown>) => apiRequest<MemoryItem[]>('/memory/search', { method: 'POST', body: payload })
export const add = (payload: Record<string, unknown>) => apiRequest<MemoryItem>('/memory/add', { method: 'POST', body: payload })
export const consolidate = (payload?: Record<string, unknown>) => apiRequest<unknown>('/memory/consolidate', { method: 'POST', body: payload || {} })
export const forget = (payload: { memory_id?: number }) => apiRequest<unknown>('/memory/forget', { method: 'POST', body: payload })
