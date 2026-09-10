import type { AgentEvent } from '../../api/types'

export function collectLiveProgress(events: AgentEvent[], runId?: number | null) {
  const seen = new Set<number>()
  const steps = new Map<string, { id: string; name: string; status: string }>()
  const blocks = new Map<string, { id: string; text: string; references: { url: string; title: string }[] }>()
  let mode = 'pending'
  let stopped = false
  let paused = false
  for (const event of events) {
    if (runId && event.run_id && event.run_id !== runId) continue
    const seq = event.event_seq || event.id
    if (seq && seen.has(seq)) continue
    if (seq) seen.add(seq)
    const p = event.payload || {}
    const kind = String(event.event_type)
    if (paused && kind === 'run_resumed') { stopped = false; paused = false }
    if (stopped) continue
    if (kind === 'interaction_mode') mode = String(p.mode)
    if (kind.startsWith('node_') && p.step_id) {
      const id = String(p.step_id)
      steps.set(id, { id, name: String(p.display_name || event.node_name), status: String(p.status || 'running') })
    }
    if (['progress_delta', 'progress_completed'].includes(kind) && p.block_id) {
      const id = `${event.run_id}:${p.block_id}`
      const previous = blocks.get(id)
      const references = Array.isArray(p.references) ? p.references.flatMap((ref) => {
        if (!ref || typeof ref !== 'object' || !('url' in ref)) return []
        const url = String(ref.url)
        return /^https?:\/\//i.test(url) ? [{ url, title: 'title' in ref ? String(ref.title) : url }] : []
      }) : previous?.references || []
      blocks.set(id, { id, text: kind === 'progress_completed' ? String(p.text || previous?.text || '')
        : (previous?.text || '') + String(p.text || ''), references })
    }
    if (['run_interrupted', 'run_failed', 'run_completed', 'run_paused'].includes(kind)) {
      stopped = true
      paused = kind === 'run_paused'
    }
  }
  return { mode, steps: [...steps.values()], blocks: [...blocks.values()], stopped }
}
