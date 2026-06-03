import { apiRequest, normalizeList } from './client'
import type { ApprovalItem } from './types'

export const list = async (params?: Record<string, unknown>) => normalizeList<ApprovalItem>(await apiRequest<unknown>('/approvals', { query: params as Record<string, string> }))
export const approve = (approvalId: number | string) => apiRequest<ApprovalItem>(`/approvals/${approvalId}/approve`, { method: 'POST' })
export const reject = (approvalId: number | string) => apiRequest<ApprovalItem>(`/approvals/${approvalId}/reject`, { method: 'POST' })
