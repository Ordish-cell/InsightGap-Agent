import { apiRequest } from './client'
import type { LlmConnection, LlmModelConfig, LlmProviderDefinition } from './types'

export const getCatalog = () => apiRequest<LlmProviderDefinition[]>('/llm/catalog')
export const getConnections = () => apiRequest<LlmConnection[]>('/llm/connections')
export const createConnection = (payload: Record<string, unknown>) => apiRequest<LlmConnection>('/llm/connections', { method: 'POST', body: payload })
export const updateConnection = (id: number, payload: Record<string, unknown>) => apiRequest<LlmConnection>(`/llm/connections/${id}`, { method: 'PATCH', body: payload })
export const deleteConnection = (id: number) => apiRequest<void>(`/llm/connections/${id}`, { method: 'DELETE' })
export const testConnection = (payload: Record<string, unknown>) => apiRequest<{ status: string; models: Array<{ model_id: string; display_name: string }> }>('/llm/connections/test', { method: 'POST', body: payload })
export const discoverModels = (id: number) => apiRequest<LlmModelConfig[]>(`/llm/connections/${id}/discover-models`, { method: 'POST' })
export const addModel = (connectionId: number, payload: Record<string, unknown>) => apiRequest<LlmModelConfig>(`/llm/connections/${connectionId}/models`, { method: 'POST', body: payload })
export const updateModel = (connectionId: number, modelId: number, payload: Record<string, unknown>) => apiRequest<LlmModelConfig>(`/llm/connections/${connectionId}/models/${modelId}`, { method: 'PATCH', body: payload })
export const deleteModel = (connectionId: number, modelId: number) => apiRequest<void>(`/llm/connections/${connectionId}/models/${modelId}`, { method: 'DELETE' })
export const getPreferences = () => apiRequest<{ default_model_config_id: number | null }>('/llm/preferences')
export const updatePreferences = (modelId: number | null) => apiRequest<{ default_model_config_id: number | null }>('/llm/preferences', { method: 'PATCH', body: { default_model_config_id: modelId } })
