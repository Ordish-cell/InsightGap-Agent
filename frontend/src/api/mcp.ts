import { apiRequest, normalizeList } from './client'
import type { McpTool, McpToolCall } from './types'

export const listTools = async () => normalizeList<McpTool>(await apiRequest<unknown>('/mcp/tools'))
export const getTool = (toolName: string) => apiRequest<McpTool>(`/mcp/tools/${toolName}`)
export const createToolCall = (payload: Record<string, unknown>) => apiRequest<McpToolCall>('/mcp/tool-calls', { method: 'POST', body: payload })
export const listToolCalls = async (params?: Record<string, unknown>) => normalizeList<McpToolCall>(await apiRequest<unknown>('/mcp/tool-calls', { query: params as Record<string, string> }))
export const getToolCall = (toolCallId: number | string) => apiRequest<McpToolCall>(`/mcp/tool-calls/${toolCallId}`)
export const health = () => apiRequest<unknown>('/mcp/health')
