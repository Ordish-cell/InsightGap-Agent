import type { ApiEnvelope, ApiListResponse } from './types'

export class ApiError extends Error {
  status: number
  details?: unknown

  constructor(status: number, message: string, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

type Query = Record<string, string | number | boolean | null | undefined>

interface RequestOptions {
  method?: string
  body?: unknown
  query?: Query
  headers?: HeadersInit
}

export function apiBaseUrl() {
  return localStorage.getItem('apiBaseUrl') || 'http://127.0.0.1:8000/api/v1'
}

export function authToken() {
  return localStorage.getItem('authToken')
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = new URL(`${apiBaseUrl()}${path}`)
  Object.entries(options.query || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') url.searchParams.set(key, String(value))
  })

  const headers = new Headers(options.headers)
  const token = authToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  let body: BodyInit | undefined
  if (options.body instanceof FormData) {
    body = options.body
  } else if (options.body !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(options.body)
  }

  const response = await fetch(url, { method: options.method || 'GET', headers, body })
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json') ? await response.json() : await response.text()

  if (!response.ok) {
    if (response.status === 401) localStorage.removeItem('authToken')
    const message =
      typeof payload === 'object' && payload?.error?.message
        ? payload.error.message
        : typeof payload === 'object' && payload?.detail
          ? String(payload.detail)
          : `HTTP ${response.status}`
    throw new ApiError(response.status, message, payload)
  }

  if (typeof payload === 'object' && payload && 'success' in payload) {
    const envelope = payload as ApiEnvelope<T>
    if (envelope.success === false) {
      throw new ApiError(response.status, envelope.error?.message || 'API request failed', envelope.error)
    }
    return envelope.data as T
  }

  return payload as T
}

export function normalizeList<T>(value: unknown): T[] {
  if (Array.isArray(value)) return value as T[]
  const list = value as ApiListResponse<T>
  if (Array.isArray(list?.items)) return list.items
  if (Array.isArray(list?.results)) return list.results
  if (Array.isArray(list?.data)) return list.data
  if (Array.isArray((value as { cards?: T[] })?.cards)) return (value as { cards: T[] }).cards
  if (Array.isArray((value as { tool_calls?: T[] })?.tool_calls)) return (value as { tool_calls: T[] }).tool_calls
  if (Array.isArray((value as { tools?: T[] })?.tools)) return (value as { tools: T[] }).tools
  return []
}
