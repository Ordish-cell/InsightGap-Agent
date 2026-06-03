import { apiRequest } from './client'
import type { HealthResponse } from './types'

export const health = () => apiRequest<HealthResponse>('/health')
export const dependencies = () => apiRequest<HealthResponse>('/health/dependencies')
