import { useState } from 'react'
import type { LlmConnection, LlmProviderDefinition } from '../../api/types'
type Props = { provider: LlmProviderDefinition; fields: Record<string, unknown>; connection?: LlmConnection; onChange: (fields: Record<string, unknown>) => void }
export function ConnectionFields({ provider, fields, connection, onChange }: Props) {
  const [showKey, setShowKey] = useState(false)
  function renderField(definition: LlmProviderDefinition['fields'][number]) {
    const value = fields[definition.key], secret = connection?.secrets?.[definition.key]
    const set = (next: unknown) => onChange({ ...fields, [definition.key]: next })
    return <label key={definition.key} className="mc-field"><span>{definition.label}{definition.required && !secret?.configured ? <small>必填</small> : null}</span>
      {definition.kind === 'select' ? <select value={String(value ?? definition.default ?? '')} onChange={event => set(event.target.value)}>{definition.options?.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select>
        : ['json', 'secret_json'].includes(definition.kind) ? <textarea rows={3} spellCheck={false} value={typeof value === 'string' ? value : value === undefined ? '' : JSON.stringify(value, null, 2)} placeholder={secret?.configured ? '已保存，留空保留' : '{"X-Token": "…"}'} onChange={event => set(event.target.value)} />
          : <div className="mc-input-action"><input type={definition.kind === 'secret' && !showKey ? 'password' : 'text'} autoComplete="off" spellCheck={false} value={String(value ?? '')} placeholder={secret?.configured ? `${secret.masked}，留空保留` : definition.placeholder || String(definition.default ?? '')} onChange={event => set(event.target.value)} />{definition.kind === 'secret' ? <button type="button" onClick={() => setShowKey(!showKey)}>{showKey ? '隐藏' : '显示'}</button> : null}</div>}
    </label>
  }
  const advanced = new Set(['auth_header', 'custom_headers', 'site_url', 'app_name'])
  return <>{provider.fields.filter(item => !advanced.has(item.key)).map(renderField)}{provider.fields.some(item => advanced.has(item.key)) ? <details className="mc-advanced"><summary>高级设置</summary>{provider.fields.filter(item => advanced.has(item.key)).map(renderField)}</details> : null}</>
}
export function serializeFields(provider: LlmProviderDefinition, fields: Record<string, unknown>) {
  const result = { ...fields }
  for (const field of provider.fields) {
    if (['json', 'secret_json'].includes(field.kind) && typeof result[field.key] === 'string') {
      const text = String(result[field.key]).trim()
      if (!text) { delete result[field.key]; continue }
      try { result[field.key] = JSON.parse(text) } catch { throw new Error(`${field.label} 必须是有效 JSON。`) }
    }
  }
  return result
}
