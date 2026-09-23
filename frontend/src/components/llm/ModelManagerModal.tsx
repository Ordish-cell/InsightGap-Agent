import { usePresence } from '../common/motion'
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import * as llm from '../../api/llm'
import type { LlmConnection, LlmProviderDefinition } from '../../api/types'
import { ConnectionFields, serializeFields } from './ConnectionFields'
import { connectionStatus, isConnectionReady, protocolLabels, readableError } from './modelUi'
import './models.css'

type Props = { open: boolean; onClose: () => void; onChanged?: () => void; initialProvider?: string }
export function ModelManagerModal(props: Props) {
  const present = usePresence(props.open)
  return present ? createPortal(<ModelManagerContent {...props} />, document.body) : null
}
function ModelManagerContent({ open, onClose, onChanged, initialProvider }: Props) {
  const dialog = useRef<HTMLDivElement>(null)
  const initialized = useRef(false)
  const [catalog, setCatalog] = useState<LlmProviderDefinition[]>([])
  const [connections, setConnections] = useState<LlmConnection[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [providerKey, setProviderKey] = useState(initialProvider || 'custom')
  const [editing, setEditing] = useState(false)
  const [displayName, setDisplayName] = useState('')
  const [protocol, setProtocol] = useState('')
  const [fields, setFields] = useState<Record<string, unknown>>({})
  const [modelId, setModelId] = useState('')
  const [modelQuery, setModelQuery] = useState('')
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [defaultId, setDefaultId] = useState<number | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const selected = connections.find(item => item.id === selectedId)
  const provider = catalog.find(item => item.key === providerKey)
  async function reload() {
    const [nextCatalog, nextConnections, preferences] = await Promise.all([llm.getCatalog(), llm.getConnections(), llm.getPreferences()])
    setCatalog(nextCatalog); setConnections(nextConnections); setDefaultId(preferences.default_model_config_id)
    return { nextCatalog, nextConnections }
  }
  function select(connection: LlmConnection) {
    setSelectedId(connection.id); setProviderKey(connection.provider); setDisplayName(connection.display_name)
    setProtocol(connection.protocol); setFields(connection.fields || {}); setModelId('')
    setModelQuery(''); setEditing(false); setError(''); setNotice(''); setConfirmDelete(false)
  }
  function create(provider: LlmProviderDefinition) {
    setSelectedId(null); setProviderKey(provider.key); setDisplayName(provider.label); setProtocol(provider.protocol)
    setFields(Object.fromEntries(provider.fields.filter(item => item.default !== undefined && item.default !== '').map(item => [item.key, item.default])))
    setModelId(''); setEditing(true); setError(''); setNotice(''); setConfirmDelete(false)
  }
  useEffect(() => {
    let active = true
    void reload().then(({ nextCatalog, nextConnections }) => {
      if (!active || initialized.current) return
      initialized.current = true
      if (nextConnections.length && !initialProvider) select(nextConnections[0]!)
      else { const preset = nextCatalog.find(item => item.key === (initialProvider || 'custom')) || nextCatalog[0]; if (preset) create(preset) }
    }).catch(exc => { if (active) setError(readableError(exc)) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    dialog.current?.focus()
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = overflow; previous?.focus() }
  }, [open])
  async function act(label: string, action: () => Promise<string | void>) {
    if (busy) return
    setBusy(label); setError(''); setNotice('')
    try { const message = await action(); await reload(); if (message) setNotice(message); onChanged?.() }
    catch (exc) { setError(readableError(exc)); await reload().catch(() => undefined); onChanged?.() }
    finally { setBusy('') }
  }
  async function save() {
    if (!provider) return
    await act('保存配置', async () => {
      const values = serializeFields(provider, fields)
      const saved = selectedId ? await llm.updateConnection(selectedId, { display_name: displayName, protocol, fields: values })
        : await llm.createConnection({ provider: provider.key, display_name: displayName, protocol, fields: values, model_id: modelId.trim() })
      select(saved)
      return isConnectionReady(saved) ? '配置已保存。' : '连接已保存。获取或添加模型后，点击“测试生成”。'
    })
  }
  async function test(model: string) {
    if (!selectedId) return
    await act('测试生成', async () => {
      const result = await llm.testConnection({ connection_id: selectedId, model_id: model })
      return `模型 ${model} 生成测试通过${result.latency_ms !== undefined ? `，耗时 ${result.latency_ms} ms` : ''}。`
    })
  }
  const filtered = connections.filter(item => `${item.display_name} ${item.provider}`.toLowerCase().includes(query.trim().toLowerCase()))
  const models = (selected?.models || []).filter(item => `${item.display_name} ${item.model_id}`.toLowerCase().includes(modelQuery.trim().toLowerCase()))
  return <div inert={!open} className={`mc-backdrop ${open ? 'is-open' : 'is-closing'}`} onMouseDown={event => { if (event.target === event.currentTarget && !busy) onClose() }}>
    <div className="mc-dialog" role="dialog" aria-modal="true" aria-labelledby="mc-title" tabIndex={-1} ref={dialog} onKeyDown={event => {
      if (event.key === 'Escape' && !busy) { event.stopPropagation(); onClose() }
      if (event.key === 'Tab') {
        const targets = Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary') || []).filter(item => item.getClientRects().length)
        const first = targets[0], last = targets.at(-1)
        if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last?.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
      }
    }}>
      <header className="mc-header"><div><h2 id="mc-title">模型连接</h2><p>管理供应商，为新会话选择默认模型。</p></div><button className="mc-icon-button" type="button" aria-label="关闭模型设置" disabled={!!busy} onClick={onClose}>×</button></header>
      <div className="mc-body"><aside className="mc-sidebar" aria-label="已保存的供应商">
        <div className="mc-sidebar-heading"><strong>我的连接 <small>{connections.length}</small></strong><button type="button" disabled={!!busy || !catalog.length} onClick={() => create(catalog.find(item => item.key === 'custom') || catalog[0]!)} aria-label="添加连接">＋</button></div>
        <input className="mc-search" aria-label="搜索连接" placeholder="搜索连接…" value={query} onChange={event => setQuery(event.target.value)} />
        <nav>{loading ? <p className="mc-hint">正在加载连接…</p> : filtered.map(connection => <button type="button" className={`mc-connection ${connection.id === selectedId ? 'selected' : ''}`} key={connection.id} disabled={!!busy} onClick={() => select(connection)} aria-current={connection.id === selectedId ? 'true' : undefined}><span className="mc-provider-mark" aria-hidden="true">{connection.display_name.slice(0, 1).toUpperCase()}</span><span><strong>{connection.display_name}</strong><small>{connectionStatus(connection)}{connection.models.some(model => model.id === defaultId) ? ' · 默认' : ''}</small></span></button>)}{!loading && !filtered.length ? <p className="mc-hint">{query ? '没有匹配的连接。' : '添加第一个模型连接，开始聊天。'}</p> : null}</nav>
        <p className="mc-sidebar-note">默认模型用于新会话。当前任务保持原模型。</p>
      </aside><main className="mc-main" aria-busy={!!busy || loading}>
        {error ? <div className="mc-feedback error" role="alert">{error}</div> : null}{notice ? <div className="mc-feedback success" role="status">{notice}</div> : null}
        {loading ? <div className="mc-loading">正在读取模型配置…</div> : provider ? <>
          <div className="mc-detail-heading"><div><h3>{selected?.display_name || '添加连接'}</h3><p>{selected ? protocolLabels[selected.protocol] : '选择预设，填写地址和密钥。'}</p></div>{selected ? <span className={`mc-status ${isConnectionReady(selected) ? 'ready' : ''}`}>{connectionStatus(selected)}</span> : null}</div>
          {selected ? <div className="mc-tabs"><button type="button" disabled={!!busy} aria-pressed={!editing} onClick={() => setEditing(false)}>模型列表</button><button type="button" disabled={!!busy} aria-pressed={editing} onClick={() => { setFields(selected.fields || {}); setDisplayName(selected.display_name); setProtocol(selected.protocol); setEditing(true); setConfirmDelete(false) }}>连接配置</button></div> : null}
          {editing ? <form onSubmit={event => { event.preventDefault(); void save() }} className="mc-form"><fieldset disabled={!!busy}>
            {!selected ? <label className="mc-field"><span>供应商预设</span><select value={providerKey} onChange={event => { const next = catalog.find(item => item.key === event.target.value); if (next) create(next) }}>{catalog.map(item => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label> : null}
            <label className="mc-field"><span>连接名称</span><input required value={displayName} onChange={event => setDisplayName(event.target.value)} placeholder="例如：日常使用 / 本地模型" /></label>
            <label className="mc-field"><span>API 协议</span><select value={protocol} onChange={event => setProtocol(event.target.value)}>{provider.protocols.map(item => <option key={item} value={item}>{protocolLabels[item] || item}</option>)}</select></label>
            <ConnectionFields key={`${providerKey}-${selectedId}`} provider={provider} fields={fields} connection={selected} onChange={setFields} />
            {!selected && provider.key !== 'azure_openai' ? <label className="mc-field"><span>首个模型 ID <small>选填</small></span><input value={modelId} onChange={event => setModelId(event.target.value)} placeholder="可手动填写，或保存后获取模型列表" /></label> : null}
            <p className="mc-hint">密钥加密保存。修改地址、协议或密钥后需重新测试生成。</p>
            <div className="mc-footer"><button type="submit" className="mc-primary">{busy === '保存配置' ? '正在保存…' : '保存连接'}</button>{selected ? <button type="button" onClick={() => { setEditing(false); setError('') }}>取消编辑</button> : null}</div>
          </fieldset>{selected ? <div className="mc-delete-area">{confirmDelete ? <><p>删除后，新任务不能再使用这个连接。已有会话记录保留。</p><button type="button" disabled={!!busy} onClick={() => void act('删除连接', async () => { await llm.deleteConnection(selected.id); create(provider); return '连接已删除。' })}>确认删除</button><button type="button" onClick={() => setConfirmDelete(false)}>取消</button></> : <button type="button" disabled={!!busy} onClick={() => setConfirmDelete(true)}>删除连接</button>}</div> : null}</form>
            : selected ? <section className="mc-models">
              <div className="mc-model-toolbar"><input aria-label="搜索模型" placeholder="搜索模型名称或 ID…" value={modelQuery} onChange={event => setModelQuery(event.target.value)} /><button type="button" disabled={!!busy} onClick={() => void act('获取模型', async () => { await llm.discoverModels(selected.id); return '模型目录已更新。请测试要使用的模型。' })}>{busy === '获取模型' ? '正在获取…' : '获取模型'}</button></div>
              <p className="mc-hint">测试生成会发送一条简短请求。模型目录可访问不代表所有模型都可调用。</p>
              {selected.last_test_error && !error ? <p className="mc-feedback error">上次测试：{readableError(selected.last_test_error)}</p> : null}
              <div className="mc-model-list">{models.map(model => <div className={`mc-model-row ${!model.enabled ? 'disabled' : ''}`} key={model.id}><div className="mc-model-name"><strong>{model.display_name}</strong><code>{model.model_id}</code><small>{!model.enabled ? '已停用' : model.id === defaultId ? '新会话默认模型' : model.source === 'discovered' ? '在线获取' : '手动配置'}</small></div><div className="mc-row-actions">
                <button type="button" disabled={!!busy || !model.enabled} onClick={() => void test(model.model_id)}>测试生成</button>
                <button type="button" className={model.id === defaultId ? 'mc-default' : ''} disabled={!!busy || !model.enabled || !isConnectionReady(selected) || model.id === defaultId} onClick={() => void act('切换默认', async () => { await llm.updatePreferences(model.id); return `新会话默认使用 ${model.display_name}。` })}>{model.id === defaultId ? '✓ 默认' : '设为默认'}</button>
                <button type="button" disabled={!!busy} aria-label={`${model.enabled ? '停用' : '启用'} ${model.display_name}`} onClick={() => void act('更新模型', async () => { if (model.enabled) await llm.deleteModel(selected.id, model.id); else await llm.updateModel(selected.id, model.id, { enabled: true }) })}>{model.enabled ? '停用' : '启用'}</button></div></div>)}</div>
              {!models.length ? <div className="mc-empty"><strong>{modelQuery ? '没有匹配的模型' : '添加你要使用的模型'}</strong><p>{modelQuery ? '尝试搜索模型 ID 或清空搜索。' : '点击“获取模型”，也可以在下方手动输入。'}</p></div> : null}
              <form className="mc-add-model" onSubmit={event => { event.preventDefault(); void act('添加模型', async () => { await llm.addModel(selected.id, { model_id: modelId.trim() }); setModelId(''); setModelQuery(''); return '模型已添加，可以测试生成。' }) }}><input aria-label="手动模型 ID" placeholder="手动输入模型 ID" required value={modelId} onChange={event => setModelId(event.target.value)} /><button disabled={!!busy || !modelId.trim()} type="submit">添加模型</button></form>
            </section> : null}
        </> : <button type="button" onClick={() => { setLoading(true); void reload().then(({ nextCatalog }) => { if (nextCatalog[0]) create(nextCatalog[0]) }).catch(exc => setError(readableError(exc))).finally(() => setLoading(false)) }}>重新加载</button>}
        {busy ? <p className="mc-busy" role="status">正在{busy}…</p> : null}
      </main></div>
    </div>
  </div>
}
