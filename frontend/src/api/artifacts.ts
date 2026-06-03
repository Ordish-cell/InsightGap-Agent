import { apiRequest, normalizeList } from './client'
import type { Artifact } from './types'

export const list = async (params?: Record<string, unknown>) => normalizeList<Artifact>(await apiRequest<unknown>('/artifacts', { query: params as Record<string, string> }))
export const get = (artifactId: number | string) => apiRequest<Artifact>(`/artifacts/${artifactId}`)
