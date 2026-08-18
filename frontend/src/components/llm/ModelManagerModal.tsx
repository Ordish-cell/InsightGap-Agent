import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import * as llm from '../../api/llm'
import type { LlmConnection, LlmProviderDefinition } from '../../api/types'

function readableError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error)
  const labels: Record<string, string> = {
    authentication_failed: '认证失败，请检查密钥或认证 Header。',
    endpoint_or_model_not_found: '端点或模型不存在，请检查地址和模型 ID。',
    connection_timeout: '连接超时，请检查网络、端点和代理设置。',
    rate_limited: '提供商正在限流，请稍后重试。',
    protocol_not_supported: '所选协议与该提供商不兼容。',
    connection_not_verified: '请先通过连接测试，再刷新模型。',
  }
  const match = Object.entries(labels).find(([code]) => message.includes(code))
  return match?.[1] || message
}

type Props = {
  open: boolean
  onClose: () => void
  onChanged?: () => void
  initialProvider?: string
}

export function ModelManagerModal({ open, onClose, onChanged, initialProvider }: Props) {
  const [catalog, setCatalog] = useState<LlmProviderDefinition[]>([])
  const [connections, setConnections] = useState<LlmConnection[]>([])
  const [selectedProvider, setSelectedProvider] = useState(initialProvider || 'openai')
  const [selectedConnectionId, setSelectedConnectionId] = useState<number | null>(null)
  const [displayName, setDisplayName] = useState('')
  const [protocol, setProtocol] = useState('')
  const [fields, setFields] = useState<Record<string, unknown>>({})
  const [modelId, setModelId] = useState('')
  const [manualModelId, setManualModelId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [revealedSecrets, setRevealedSecrets] = useState<Set<string>>(new Set())
  const [defaultModelId, setDefaultModelId] = useState<number | null>(null)

  const provider = useMemo(() => catalog.find((item) => item.key === selectedProvider), [catalog, selectedProvider])
  const selectedConnection = connections.find((item) => item.id === selectedConnectionId)

  async function reload() {
    const [nextCatalog, nextConnections, preferences] = await Promise.all([llm.getCatalog(), llm.getConnections(), llm.getPreferences()])
    setCatalog(nextCatalog)
    setConnections(nextConnections)
    setDefaultModelId(preferences.default_model_config_id)
  }

  useEffect(() => {
    if (!open) return
    if (initialProvider) {
      setSelectedConnectionId(null)
      setSelectedProvider(initialProvider)
    }
    setError('')
    setRevealedSecrets(new Set())
    void reload().catch((exc) => setError(readableError(exc)))
  }, [open, initialProvider])

  useEffect(() => {
    if (!open) return
    const handleKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open, onClose])

  useEffect(() => {
    if (!provider || selectedConnection) return
    setProtocol(provider.protocol)
    setDisplayName(provider.label)
    setFields(Object.fromEntries(provider.fields.filter((item) => item.default !== undefined && item.default !== '').map((item) => [item.key, item.default])))
    setModelId(provider.models[1]?.model_id || provider.models[0]?.model_id || '')
  }, [provider, selectedConnection])

  function selectConnection(connection: LlmConnection) {
    setSelectedConnectionId(connection.id)
    setSelectedProvider(connection.provider)
    setDisplayName(connection.display_name)
    setProtocol(connection.protocol)
    setFields(connection.fields || {})
    setModelId(connection.models[0]?.model_id || '')
    setError('')
    setNotice('')
  }

  function newConnection(providerKey: string) {
    setSelectedConnectionId(null)
    setSelectedProvider(providerKey)
    setDisplayName('')
    setFields({})
    setModelId('')
    setError('')
    setNotice('')
  }

  async function save(testAfterSave: boolean) {
    if (!provider) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const payload = { provider: provider.key, protocol, display_name: displayName || provider.label, fields, model_id: modelId }
      const saved = selectedConnectionId
        ? await llm.updateConnection(selectedConnectionId, { protocol, display_name: displayName, fields })
        : await llm.createConnection(payload)
      setSelectedConnectionId(saved.id)
      if (testAfterSave) {
        await llm.testConnection({ connection_id: saved.id, model_id: modelId })
        setNotice('连接测试通过，可以在输入栏使用。')
      } else {
        setNotice('草稿已保存，测试通过后才能用于任务。')
      }
      await reload()
      onChanged?.()
    } catch (exc) {
      setError(readableError(exc))
      await reload().catch(() => undefined)
    } finally {
      setBusy(false)
    }
  }

  async function refreshModels() {
    if (!selectedConnectionId) return
    setBusy(true)
    setError('')
    try {
      await llm.discoverModels(selectedConnectionId)
      await reload()
      setNotice('模型列表已刷新。')
      onChanged?.()
    } catch (exc) {
      setError(readableError(exc))
    } finally {
      setBusy(false)
    }
  }

  async function addManualModel() {
    if (!selectedConnectionId || !manualModelId.trim()) return
    setBusy(true)
    try {
      await llm.addModel(selectedConnectionId, { model_id: manualModelId.trim(), display_name: manualModelId.trim(), source: 'manual' })
      setManualModelId('')
      await reload()
      onChanged?.()
    } catch (exc) {
      setError(readableError(exc))
    } finally {
      setBusy(false)
    }
  }

  async function removeConnection() {
    if (!selectedConnectionId || !window.confirm('删除这个模型连接？已启动的任务仍会按原模型快照继续，新的任务不能再选择它。')) return
    setBusy(true)
    try {
      await llm.deleteConnection(selectedConnectionId)
      setSelectedConnectionId(null)
      await reload()
      onChanged?.()
    } finally {
      setBusy(false)
    }
  }

  async function removeModel(modelPk: number) {
    if (!selectedConnectionId) return
    setBusy(true)
    setError('')
    try {
      await llm.deleteModel(selectedConnectionId, modelPk)
      await reload()
      onChanged?.()
    } catch (exc) {
      setError(readableError(exc))
    } finally {
      setBusy(false)
    }
  }

  async function makeDefault(modelPk: number) {
    setBusy(true)
    setError('')
    try {
      await llm.updatePreferences(modelPk)
      setDefaultModelId(modelPk)
      setNotice('默认模型已更新。')
      onChanged?.()
    } catch (exc) {
      setError(readableError(exc))
    } finally {
      setBusy(false)
    }
  }

  if (!open) return null

  return createPortal(
    <div className="model-modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="model-manager-dialog" role="dialog" aria-modal="true" aria-labelledby="model-manager-title">
        <header className="model-manager-header">
          <div><h2 id="model-manager-title">模型与提供商</h2><p>保存连接信息，测试通过后选择模型。</p></div>
          <button type="button" className="model-close-button" onClick={onClose} aria-label="关闭模型设置">×</button>
        </header>
        <div className="model-manager-body">
          <aside className="model-provider-list" aria-label="模型提供商">
            {connections.map((connection) => (
              <button key={connection.id} type="button" className={selectedConnectionId === connection.id ? 'active' : ''} onClick={() => selectConnection(connection)}>
                <span>{connection.display_name}</span><small>{connection.last_test_status === 'passed' ? '已验证' : '待验证'}</small>
              </button>
            ))}
            <div className="model-provider-divider">添加连接</div>
            {catalog.map((item) => <button key={item.key} type="button" className={!selectedConnectionId && selectedProvider === item.key ? 'active' : ''} onClick={() => newConnection(item.key)}>{item.label}</button>)}
          </aside>
          <div className="model-config-form">
            {provider ? <>
              <div className="model-form-row two"><label>连接名称<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder={provider.label} /></label><label>API 协议<select value={protocol || provider.protocol} onChange={(event) => setProtocol(event.target.value)}>{provider.protocols.map((item) => <option key={item} value={item}>{item}</option>)}</select></label></div>
              <div className="model-form-fields">
                {provider.fields.map((definition) => {
                  const value = fields[definition.key]
                  const secret = selectedConnection?.secrets?.[definition.key]
                  return <label key={definition.key}>{definition.label}{definition.required ? <span className="required-mark">必填</span> : null}
                    {definition.kind === 'select' ? <select value={String(value ?? definition.default ?? '')} onChange={(event) => setFields((current) => ({ ...current, [definition.key]: event.target.value }))}>{(definition.options || []).map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select> : definition.kind === 'json' || definition.kind === 'secret_json' ? <textarea rows={3} value={definition.kind === 'secret_json' && value === undefined ? '' : typeof value === 'string' ? value : JSON.stringify(value ?? {}, null, 2)} placeholder={secret?.configured ? `${secret.masked}，留空保留` : undefined} onChange={(event) => { try { setFields((current) => ({ ...current, [definition.key]: JSON.parse(event.target.value) })) } catch { setFields((current) => ({ ...current, [definition.key]: event.target.value })) } }} /> : definition.kind === 'secret' ? <span className="model-secret-field"><input type={revealedSecrets.has(definition.key) ? 'text' : 'password'} value={String(value ?? '')} placeholder={secret?.configured ? `${secret.masked}，留空保留` : definition.placeholder || String(definition.default ?? '')} onChange={(event) => setFields((current) => ({ ...current, [definition.key]: event.target.value }))} /><button type="button" onClick={() => setRevealedSecrets((current) => { const next = new Set(current); next.has(definition.key) ? next.delete(definition.key) : next.add(definition.key); return next })}>{revealedSecrets.has(definition.key) ? '隐藏' : '显示'}</button></span> : <input type={definition.kind === 'url' ? 'url' : 'text'} value={String(value ?? '')} placeholder={definition.placeholder || String(definition.default ?? '')} onChange={(event) => setFields((current) => ({ ...current, [definition.key]: event.target.value }))} />}
                  </label>
                })}
              </div>
              {!selectedConnectionId && provider.key === 'custom' ? <label>模型 ID<span className="required-mark">必填</span><input value={modelId} onChange={(event) => setModelId(event.target.value)} placeholder="例如 my-model-v1" /></label> : null}
              {!selectedConnectionId && provider.key !== 'custom' && provider.key !== 'azure_openai' ? <label>首个模型<select value={modelId} onChange={(event) => setModelId(event.target.value)}><option value="">测试后在线发现</option>{provider.models.map((model) => <option key={model.model_id} value={model.model_id}>{model.display_name}</option>)}</select></label> : null}
              {selectedConnection ? <div className="configured-models"><div className="configured-models-header"><h3>可用模型</h3><button type="button" onClick={refreshModels} disabled={busy || selectedConnection.status !== 'active'}>刷新模型</button></div>{selectedConnection.models.length ? <div className="model-chip-list">{selectedConnection.models.map((model) => <span key={model.id}>{model.display_name}<small>{model.source}{model.capabilities.tools ? ' · 工具' : ''}{model.capabilities.structured_output ? ' · JSON' : ''}{model.capabilities.streaming ? ' · 流式' : ''}</small><button type="button" className="model-default-action" onClick={() => void makeDefault(model.id)} disabled={busy || !model.enabled || selectedConnection.status !== 'active'}>{defaultModelId === model.id ? '默认' : '设默认'}</button><button type="button" aria-label={`停用 ${model.display_name}`} onClick={() => void removeModel(model.id)} disabled={busy}>×</button></span>)}</div> : <p>还没有模型，刷新目录或手动添加。</p>}<div className="manual-model-row"><input value={manualModelId} onChange={(event) => setManualModelId(event.target.value)} placeholder="手动输入模型 ID" /><button type="button" onClick={addManualModel} disabled={busy || !manualModelId.trim()}>添加模型</button></div></div> : null}
              {error ? <div className="model-form-message error" role="alert">{error}</div> : null}{notice ? <div className="model-form-message success" role="status">{notice}</div> : null}
              <footer className="model-manager-actions">{selectedConnectionId ? <button type="button" className="danger-text-button" onClick={removeConnection} disabled={busy}>删除连接</button> : <span />}<div><button type="button" onClick={() => save(false)} disabled={busy}>保存草稿</button><button type="button" className="primary" onClick={() => save(true)} disabled={busy}>{busy ? '正在检查…' : '保存并测试'}</button></div></footer>
            </> : <p>正在读取提供商目录…</p>}
          </div>
        </div>
      </section>
    </div>, document.body,
  )
}
