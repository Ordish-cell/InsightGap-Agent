import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import * as llm from '../../api/llm'
import type { LlmConnection, LlmModelConfig } from '../../api/types'
import { ModelManagerModal } from './ModelManagerModal'
import { availableModels, readableError } from './modelUi'

type Props = { value: number | null; onChange: (value: number | null) => void; disabled?: boolean }

export function ModelSelector({ value, onChange, disabled }: Props) {
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [connections, setConnections] = useState<LlmConnection[]>([])
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const [open, setOpen] = useState(false)
  const [managerOpen, setManagerOpen] = useState(false)
  const [position, setPosition] = useState({ left: 0, bottom: 0 })
  const [defaultModelId, setDefaultModelId] = useState<number | null>(null)
  const [loaded, setLoaded] = useState(false)

  async function reload() {
    const [nextConnections, preferences] = await Promise.all([llm.getConnections(), llm.getPreferences()])
    setConnections(nextConnections)
    setError('')
    setDefaultModelId(preferences.default_model_config_id)
    setLoaded(true)
  }

  useEffect(() => { void reload().catch(exc => setError(readableError(exc))) }, [])
  useEffect(() => {
    if (!open) return
    const handleKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') { setOpen(false); buttonRef.current?.focus() } }
    const handleResize = () => setOpen(false)
    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('resize', handleResize)
    return () => { window.removeEventListener('keydown', handleKeyDown); window.removeEventListener('resize', handleResize) }
  }, [open])

  const available = availableModels(connections)
  const filtered = availableModels(connections, query)
  const selected = available.find((item) => item.model.id === value)
  const availableKey = available.map((item) => item.model.id).join(',')

  useEffect(() => {
    // Never silently replace an explicit conversation choice with another provider.
    if (!loaded || disabled || value !== null) return
    const fallback = available.find((item) => item.model.id === defaultModelId)
    const next = fallback?.model.id ?? null
    if (next !== value) onChange(next)
  }, [loaded, value, defaultModelId, availableKey, onChange, disabled])

  function toggle() {
    if (disabled) return
    const rect = buttonRef.current?.getBoundingClientRect()
    if (rect) setPosition({ left: Math.max(8, Math.min(rect.left, window.innerWidth - 352)), bottom: Math.max(8, window.innerHeight - rect.top + 8) })
    if (!open) { setQuery(''); void reload().catch(exc => setError(readableError(exc))) }
    setOpen((current) => !current)
  }

  function choose(model: LlmModelConfig) {
    onChange(model.id)
    setOpen(false)
  }

  return <>
    <button ref={buttonRef} type="button" className="composer-model-button" title={selected ? `${selected.connection.display_name} · ${selected.model.model_id}` : '选择模型连接'} onClick={toggle} disabled={disabled} aria-haspopup="dialog" aria-expanded={open}><span className={selected ? 'model-status-dot ready' : 'model-status-dot'} />{selected?.model.display_name || (value ? '模型不可用，请重选' : '选择模型')}<span aria-hidden="true">⌄</span></button>
    {open ? createPortal(<><button type="button" className="model-selector-scrim" aria-label="关闭模型选择" onClick={() => setOpen(false)} /><div className="model-selector-popover mc-picker" style={{ left: position.left, bottom: position.bottom }} role="dialog" aria-label="选择本次使用的模型">
      <div className="model-selector-heading"><strong>本次任务使用</strong><small>切换只影响下一次发送</small></div>
      <input autoFocus className="mc-search" aria-label="搜索供应商或模型" placeholder="搜索供应商或模型…" value={query} onChange={event => setQuery(event.target.value)} />
      {error ? <p className="mc-feedback error" role="alert">{error}</p> : null}
      <div className="mc-picker-results">{filtered.length ? connections.map(connection => {
        const items = filtered.filter(item => item.connection.id === connection.id)
        return items.length ? <section key={connection.id} aria-label={connection.display_name}><h4>{connection.display_name}</h4>{items.map(({ model }) => <button key={model.id} type="button" aria-pressed={model.id === value} onClick={() => choose(model)}><span><strong>{model.display_name}</strong><small>{model.model_id}{model.id === defaultModelId ? ' · 默认' : ''}</small></span>{model.id === value ? <span>✓</span> : null}</button>)}</section> : null
      }) : <div className="model-selector-empty">{query ? '没有匹配的模型。' : '尚无可用模型，请添加连接并测试生成。'}</div>}</div>
      <div className="model-selector-divider" />
      <button type="button" className="manage-models-button" onClick={() => { setOpen(false); setManagerOpen(true) }}>管理模型连接 <span>↗</span></button>
    </div></>, document.body) : null}
    <ModelManagerModal open={managerOpen} onClose={() => setManagerOpen(false)} onChanged={() => void reload().catch(exc => setError(readableError(exc)))} />
  </>
}
