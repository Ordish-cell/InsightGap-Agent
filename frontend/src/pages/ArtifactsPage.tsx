import { useEffect, useRef, useState } from 'react'
import { ActionNotice } from '../components/common/ActionNotice'
import { MarkdownRenderer } from '../components/common/MarkdownRenderer'
import { useListMotion } from '../components/common/motion'
import { useSearchParams } from 'react-router-dom'

import * as artifacts from '../api/artifacts'
import type { Artifact } from '../api/types'
import { EmptyState } from '../components/common/EmptyState'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

function previewContent(value: unknown) {
  if (typeof value === 'string' && value.trim()) return value
  return '选择左侧成果后，正文会展示在这里。'
}

export function ArtifactsPage() {
  const [params] = useSearchParams()
  const [items, setItems] = useState<Artifact[]>([])
  const [selected, setSelected] = useState<Artifact | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [readerError, setReaderError] = useState('')
  const [reading, setReading] = useState(false)
  const sequence = useRef(0)
  const listMotion = useListMotion()

  async function select(id: string | number) {
    const request = ++sequence.current
    setReading(true); setReaderError('')
    try {
      const item = await artifacts.get(id)
      if (request === sequence.current) setSelected(item)
    } catch (exc) {
      if (request === sequence.current) setReaderError(exc instanceof Error ? exc.message : '正文加载失败，请重新选择。')
    } finally { if (request === sequence.current) setReading(false) }
  }

  useEffect(() => {
    let active = true
    setLoading(true); setError('')
    artifacts.list().then(async (list) => {
      if (!active) return
      setItems(list)
      const selectedId = params.get('artifactId') || list[0]?.id
      if (selectedId) await select(selectedId)
    }).catch((exc) => { if (active) setError(exc instanceof Error ? exc.message : '成果加载失败') }).finally(() => { if (active) setLoading(false) })
    return () => { active = false; sequence.current++ }
  }, [params])

  return (
    <section className="workbench-page artifacts-page">
      <PageHeader title="成果库" description="研究、工具和智能体生成的本地成果都会沉淀到这里。" />
      {error ? <ErrorState message={error} /> : loading ? <LoadingState title="正在加载成果" /> : !items.length ? <EmptyState title="暂无成果" description="完成一次深度研究后，报告会出现在这里。" /> : (
        <div className="artifact-layout">
          <aside className="artifact-list-panel"><div className="section-title"><strong>全部成果</strong><span>{items.length} 个</span></div><div className="artifact-list" ref={listMotion}>{items.map((item) => <button className={selected?.id === item.id ? 'artifact-list-item active' : 'artifact-list-item'} key={item.id} data-entry={item.id} aria-pressed={selected?.id === item.id} onClick={() => void select(item.id)}><strong>{item.title}</strong><span><StatusPill value={item.artifact_type} /> 本地成果</span></button>)}</div></aside>
          <article className="artifact-reader" aria-busy={reading}><div className="artifact-reader-head"><div><span className="eyebrow">成果阅读</span><h2>{selected?.title || '选择一个成果'}</h2></div>{selected?.artifact_type ? <StatusPill value={selected.artifact_type} /> : null}</div><ActionNotice message={readerError} error />{reading ? <LoadingState title="正在加载正文" /> : /markdown|report/.test(selected?.artifact_type || '') ? <MarkdownRenderer content={previewContent(selected?.content)} /> : <pre className="reader-block">{previewContent(selected?.content)}</pre>}</article>
        </div>
      )}
    </section>
  )
}
