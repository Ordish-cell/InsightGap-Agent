import { apiRequest, normalizeList } from './client'
import type { FeedCard } from './types'

export const refresh = () => apiRequest<unknown>('/feed/refresh', { method: 'POST' })
export const listCards = async (params?: Record<string, unknown>) => normalizeList<FeedCard>(await apiRequest<unknown>('/feed/cards', { query: params as Record<string, string> }))
export const getCard = (cardId: number | string) => apiRequest<FeedCard>(`/feed/cards/${cardId}`)
export const feedback = (cardId: number | string, payload: Record<string, unknown>) => apiRequest<unknown>(`/feed/cards/${cardId}/feedback`, { method: 'POST', body: payload })
export const startResearch = (cardId: number | string) => apiRequest<unknown>(`/feed/cards/${cardId}/research`, { method: 'POST', body: {} })
export const sources = () => apiRequest<unknown>('/feed/sources')
export const stats = () => apiRequest<unknown>('/feed/stats')
