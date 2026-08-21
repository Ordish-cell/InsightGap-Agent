import { apiBaseUrl, authToken } from './client'
import type { ChatAttachment } from './types'

export type { ChatAttachment } from './types'

export function uploadChatAttachment(
  file: File,
  onProgress?: (progress: number) => void,
): Promise<ChatAttachment> {
  return new Promise((resolve, reject) => {
    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${apiBaseUrl()}/documents/chat-upload`, true)

    const token = authToken()
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    }

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return
      const progress = Math.round((event.loaded / event.total) * 100)
      onProgress?.(progress)
    }

    xhr.onload = () => {
      try {
        const response = JSON.parse(xhr.responseText)
        if (xhr.status >= 200 && xhr.status < 300 && response.success) {
          resolve(response.data as ChatAttachment)
        } else {
          const message = response?.error?.message || response?.message || '上传失败'
          reject(new Error(message))
        }
      } catch {
        reject(new Error('上传响应解析失败'))
      }
    }

    xhr.onerror = () => reject(new Error('网络错误，上传失败'))
    xhr.onabort = () => reject(new Error('上传已取消'))

    xhr.send(formData)
  })
}

export async function fetchDocumentStatus(documentId: number): Promise<ChatAttachment> {
  const headers = new Headers()
  const token = authToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(`${apiBaseUrl()}/documents/${documentId}/status`, { headers })
  const payload = await response.json()
  if (!response.ok || !payload.success) {
    throw new Error(payload?.error?.message || '读取文档处理状态失败')
  }
  return payload.data as ChatAttachment
}

export async function waitForDocumentReady(
  documentId: number,
  onStatus?: (status: ChatAttachment) => void,
): Promise<ChatAttachment> {
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const status = await fetchDocumentStatus(documentId)
    onStatus?.(status)
    if (status.status === 'ready' || ['ingested', 'completed'].includes(status.ingest_status || '')) return status
    if (['failed', 'interrupted'].includes(status.status || '') || status.ingest_status === 'failed') {
      throw new Error(status.error || '文档解析失败，请重新处理')
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1000))
  }
  throw new Error('文档处理超时，请稍后重试')
}

export async function retryDocumentIngest(documentId: number): Promise<ChatAttachment> {
  const headers = new Headers({ 'Content-Type': 'application/json' })
  const token = authToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(`${apiBaseUrl()}/documents/${documentId}/ingest-background`, { method: 'POST', headers })
  const payload = await response.json()
  if (!response.ok || !payload.success) throw new Error(payload?.error?.message || '重新处理失败')
  return payload.data as ChatAttachment
}

export function toApiUrl(path: string | undefined | null): string {
  if (!path) return ''
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  const base = apiBaseUrl()
  const origin = base.replace(/\/api\/v1.*$/, '')
  return origin + path
}

export async function fetchDocumentBlobUrl(documentId: number): Promise<string> {
  const url = `${apiBaseUrl()}/documents/${documentId}/file`
  const headers = new Headers()
  const token = authToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(url, { headers })
  if (!response.ok) {
    throw new Error(`Failed to load document file: ${response.status}`)
  }
  const blob = await response.blob()
  return URL.createObjectURL(blob)
}
