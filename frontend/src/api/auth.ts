import { apiRequest } from './client'
import type { AuthResponse, CurrentUser } from './types'

export const register = (payload: { email: string; password: string; nickname?: string }) => apiRequest<CurrentUser>('/auth/register', { method: 'POST', body: payload })
export const login = (payload: { email: string; password: string }) => apiRequest<AuthResponse>('/auth/login', { method: 'POST', body: payload })
export const me = () => apiRequest<CurrentUser>('/auth/me')
