import { ActionNotice } from '../components/common/ActionNotice'
import { useAction } from '../components/common/useAction'
import { FormEvent, useEffect, useState } from 'react'

import { me } from '../api/auth'
import { getConnections } from '../api/llm'
import type { CurrentUser, LlmConnection } from '../api/types'
import { JsonBlock } from '../components/common/JsonBlock'
import { PageHeader } from '../components/common/PageHeader'
import { ModelManagerModal } from '../components/llm/ModelManagerModal'

export function SettingsPage() {
  const action = useAction()
  const [connectionError, setConnectionError] = useState('')
  const [apiBaseUrl, setApiBaseUrl] = useState(localStorage.getItem('apiBaseUrl') || 'http://127.0.0.1:8000/api/v1')
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [connections, setConnections] = useState<LlmConnection[]>([])
  const [managerOpen, setManagerOpen] = useState(false)

  function loadConnections() { void getConnections().then(items => { setConnections(items); setConnectionError('') }).catch(exc => setConnectionError(exc instanceof Error ? exc.message : '模型连接加载失败')) }
  useEffect(() => { me().then(setUser).catch(() => setUser(null)); loadConnections() }, [])
  function save(event: FormEvent) { event.preventDefault(); void action.run(async () => { const url = new URL(apiBaseUrl); if (!['http:', 'https:'].includes(url.protocol)) throw new Error('请输入有效的 HTTP 或 HTTPS 服务地址。'); localStorage.setItem('apiBaseUrl', apiBaseUrl.trim()) }, '服务地址已保存。') }

  const activeCount = connections.filter((item) => item.status === 'active' && item.last_test_status === 'passed').length

  return (
    <section className="workbench-page">
      <PageHeader title="设置" description="管理模型连接、运行地址和账号信息。" />
      <ActionNotice message={action.message} error={action.failed} />
      <ActionNotice message={connectionError} error />
      <div className="settings-layout">
        <div className="stack">
          <section className="settings-section model-settings-summary">
            <div><h2>模型连接</h2><p>管理多个供应商、测试模型，并选择新会话默认使用的模型。</p></div>
            <div className="model-connection-summary"><strong>{connectionError ? '—' : activeCount}</strong><span>个可用连接</span><small>{connections.length} 个连接已保存</small></div>
            <button className="button" type="button" onClick={() => setManagerOpen(true)}>管理模型连接</button>
          </section>
          <form className="settings-section stack" onSubmit={save}>
            <div><h2>服务地址</h2><p>只影响当前浏览器访问的后端 API。</p></div>
            <label>API Base URL<input className="input" value={apiBaseUrl} onChange={(event) => setApiBaseUrl(event.target.value)} /></label>
            <button className="button" disabled={action.busy} style={{ width: 'fit-content' }}>保存服务地址</button>
          </form>
        </div>
        <aside className="settings-section current-user-section"><h2>当前用户</h2><p>{user?.email || '未读取到用户信息'}</p><p className="muted">模型配置按用户隔离。读接口只返回密钥掩码，不返回明文。</p><details><summary>技术详情</summary><JsonBlock value={user} /></details></aside>
      </div>
      <ModelManagerModal open={managerOpen} onClose={() => setManagerOpen(false)} onChanged={loadConnections} />
    </section>
  )
}
