import { apiRequest } from './client'
import type { LongTermMemoryItem, LongTermMemoryListResponse, MemoryItem, MemorySummary } from './types'

export const summary = () => apiRequest<MemorySummary>('/memory/summary')
export const search = (payload: Record<string, unknown>) => apiRequest<MemoryItem[]>('/memory/search', { method: 'POST', body: payload })
export const add = (payload: Record<string, unknown>) => apiRequest<MemoryItem>('/memory/add', { method: 'POST', body: payload })
export const consolidate = (payload?: Record<string, unknown>) => apiRequest<unknown>('/memory/consolidate', { method: 'POST', body: payload || {} })
export const forget = (payload: { memory_id?: number }) => apiRequest<unknown>('/memory/forget', { method: 'POST', body: payload })

export const listLongTermMemories = (params: Record<string, unknown>) => apiRequest<LongTermMemoryListResponse>('/memory/long-term', { query: params })
export const searchLongTermMemories = (payload: Record<string, unknown>) => apiRequest<LongTermMemoryListResponse>('/memory/long-term/search', { method: 'POST', body: payload })
export const updateMemory = (id: number, body: Record<string, unknown>) => apiRequest<LongTermMemoryItem>(`/memory/${id}`, { method: 'PATCH', body })
export const deleteMemory = (id: number) => apiRequest<{ deleted: boolean; memory_id: number }>(`/memory/${id}`, { method: 'DELETE' })
export const archiveMemory = (id: number) => apiRequest<{ memory_id: number; status: string }>(`/memory/${id}/archive`, { method: 'POST' })
export const restoreMemory = (id: number) => apiRequest<{ memory_id: number; status: string }>(`/memory/${id}/restore`, { method: 'POST' })
