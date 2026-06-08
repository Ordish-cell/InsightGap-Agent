import { apiRequest, normalizeList } from './client'
import { normalizeFeedCard } from './normalizers'
import type { FeedCard } from './types'

export const refresh = () => apiRequest<unknown>('/feed/refresh', { method: 'POST' })
export const listCards = async (params?: Record<string, unknown>) => normalizeList<FeedCard>(await apiRequest<unknown>('/feed/cards', { query: params as Record<string, string> })).map(normalizeFeedCard)
export const homeCards = async () => {
  const result = await apiRequest<{ cards?: FeedCard[]; is_complete?: boolean; error?: string; message?: string }>('/feed/home')
  return { ...result, cards: (result.cards || []).map(normalizeFeedCard) }
}
export const getCard = async (cardId: number | string) => normalizeFeedCard(await apiRequest<FeedCard>(`/feed/cards/${cardId}`))
export const feedback = (cardId: number | string, payload: Record<string, unknown>) => apiRequest<unknown>(`/feed/cards/${cardId}/feedback`, { method: 'POST', body: payload })
export const startResearch = async (cardId: number | string): Promise<{ id: string }> => {
  const result = await apiRequest<{ id?: string }>(`/feed/cards/${cardId}/research`, { method: 'POST', body: {} })
  return { id: String(result?.id || '') }
}
export const sources = () => apiRequest<unknown>('/feed/sources')
export const stats = () => apiRequest<unknown>('/feed/stats')
