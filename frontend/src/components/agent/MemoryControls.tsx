import { useEffect, useState } from 'react'
import { apiRequest } from '../../api/client'

interface MemorySettings {
  use_memory: boolean
  generate_memory: boolean
}

export function MemoryControls({ conversationId }: { conversationId?: string }) {
  const [settings, setSettings] = useState<MemorySettings | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const url = conversationId ? `/agent/conversations/${encodeURIComponent(conversationId)}/memory-settings` : '/profile/me'
  useEffect(() => {
    let active = true
    setSettings(null)
    setError('')
    apiRequest<MemorySettings>(url).then(value => { if (active) setSettings(value) }).catch(exc => {
      if (active) setError(exc instanceof Error ? exc.message : '记忆设置加载失败')
    })
    return () => { active = false }
  }, [url])

  async function update(body: Partial<Record<keyof MemorySettings, boolean | null>>) {
    setBusy(true)
    setError('')
    try {
      setSettings(await apiRequest<MemorySettings>(url, { method: conversationId ? 'PATCH' : 'PUT', body }))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '记忆设置保存失败')
    } finally {
      setBusy(false)
    }
  }

  return <div className="memory-controls" aria-label={conversationId ? '本会话记忆设置' : '全局记忆设置'} style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
    <label><input type="checkbox" checked={settings?.use_memory ?? false} disabled={!settings || busy} onChange={e => void update({ use_memory: e.target.checked })} />{conversationId ? '本会话使用记忆' : '使用已有记忆'}</label>
    <label><input type="checkbox" checked={settings?.generate_memory ?? false} disabled={!settings || busy} onChange={e => void update({ generate_memory: e.target.checked })} />{conversationId ? '本会话自动整理' : '自动整理后续对话'}</label>
    {conversationId && <button type="button" className="button ghost small" disabled={busy || !settings} onClick={() => void update({ use_memory: null, generate_memory: null })}>使用全局设置</button>}
    {error && <span role="alert">{error}</span>}
  </div>
}
