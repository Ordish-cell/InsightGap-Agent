import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import * as llm from '../../api/llm'
import type { LlmConnection, LlmModelConfig, LlmProviderDefinition } from '../../api/types'
import { ModelManagerModal } from './ModelManagerModal'

type Props = { value: number | null; onChange: (value: number | null) => void; disabled?: boolean }

export function ModelSelector({ value, onChange, disabled }: Props) {
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [connections, setConnections] = useState<LlmConnection[]>([])
  const [catalog, setCatalog] = useState<LlmProviderDefinition[]>([])
  const [open, setOpen] = useState(false)
  const [managerOpen, setManagerOpen] = useState(false)
  const [managerProvider, setManagerProvider] = useState<string | undefined>()
  const [position, setPosition] = useState({ left: 0, bottom: 0 })
  const [defaultModelId, setDefaultModelId] = useState<number | null>(null)
  const [loaded, setLoaded] = useState(false)

  async function reload() {
    const [nextConnections, nextCatalog, preferences] = await Promise.all([llm.getConnections(), llm.getCatalog(), llm.getPreferences()])
    setConnections(nextConnections)
    setCatalog(nextCatalog)
    setDefaultModelId(preferences.default_model_config_id)
    setLoaded(true)
    if (value === null && preferences.default_model_config_id) onChange(preferences.default_model_config_id)
  }

  useEffect(() => { void reload().catch(() => undefined) }, [])
  useEffect(() => {
    if (value === null && defaultModelId) onChange(defaultModelId)
  }, [value, defaultModelId, onChange])
  useEffect(() => {
    if (!open) return
    const handleKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open])

  const available = connections.flatMap((connection) => connection.status === 'active' && connection.last_test_status === 'passed' ? connection.models.filter((model) => model.enabled).map((model) => ({ connection, model })) : []).sort((left, right) => Number(right.model.id === value) - Number(left.model.id === value) || Number(right.model.id === defaultModelId) - Number(left.model.id === defaultModelId))
  const selected = available.find((item) => item.model.id === value)
  const availableKey = available.map((item) => item.model.id).join(',')

  useEffect(() => {
    if (!loaded || value === null || available.some((item) => item.model.id === value)) return
    const fallback = available.find((item) => item.model.id === defaultModelId) || available[0]
    onChange(fallback?.model.id ?? null)
  }, [loaded, value, defaultModelId, availableKey, onChange])

  function toggle() {
    if (disabled) return
    const rect = buttonRef.current?.getBoundingClientRect()
    if (rect) setPosition({ left: Math.min(rect.left, window.innerWidth - 320), bottom: window.innerHeight - rect.top + 8 })
    setOpen((current) => !current)
  }

  function choose(model: LlmModelConfig) {
    onChange(model.id)
    setOpen(false)
  }

  return <>
    <button ref={buttonRef} type="button" className="composer-model-button" onClick={toggle} disabled={disabled} aria-haspopup="listbox" aria-expanded={open}><span className={selected ? 'model-status-dot ready' : 'model-status-dot'} />{selected?.model.display_name || '选择模型'}<span aria-hidden="true">⌄</span></button>
    {open ? createPortal(<><button type="button" className="model-selector-scrim" aria-label="关闭模型选择" onClick={() => setOpen(false)} /><div className="model-selector-popover" style={{ left: position.left, bottom: position.bottom }} role="listbox">
      <div className="model-selector-heading"><strong>本次任务使用</strong><small>切换只影响下一次发送</small></div>
      {available.length ? available.map(({ connection, model }) => <button key={model.id} type="button" role="option" aria-selected={model.id === value} onClick={() => choose(model)}><span><strong>{model.display_name}</strong><small>{connection.display_name}{model.id === defaultModelId ? ' · 默认' : ''}</small></span>{model.id === value ? <span>✓</span> : null}</button>) : <div className="model-selector-empty">还没有验证通过的模型。</div>}
      <div className="model-selector-divider" />
      {catalog.map((provider) => <button key={provider.key} type="button" onClick={() => { setOpen(false); setManagerProvider(provider.key); setManagerOpen(true) }}><span><strong>{provider.label}</strong><small>{provider.models[1]?.display_name || '配置连接'}</small></span><span>＋</span></button>)}
      <button type="button" className="manage-models-button" onClick={() => { setOpen(false); setManagerProvider(undefined); setManagerOpen(true) }}>添加或管理模型</button>
    </div></>, document.body) : null}
    <ModelManagerModal open={managerOpen} initialProvider={managerProvider} onClose={() => setManagerOpen(false)} onChanged={() => void reload()} />
  </>
}
