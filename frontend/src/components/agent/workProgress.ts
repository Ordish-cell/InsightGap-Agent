import type { AgentChatMessage, AgentEvent } from '../../api/types'

export type WorkBlock = { id: string; kind: 'text' | 'tool'; text: string; role: string; status: string }

/** One ordered projection for live events and persisted replay. Classification never moves a block. */
export function projectWorkProgress(message: AgentChatMessage, events: AgentEvent[], answer: string) {
  const blocks = new Map<string, WorkBlock>()
  const steps = new Map<string, WorkBlock>()
  const seen = new Set<number>()
  const concreteToolSteps = new Set<string>()
  let terminal = false, paused = false, latestText = '', lastStep = ''
  const ordered = [...events].sort((a, b) => (a.event_seq || a.id || 0) - (b.event_seq || b.id || 0))
  for (const event of ordered) {
    if (Number(event.run_id) !== Number(message.run_id)) continue
    const p = event.payload || {}, kind = String(event.event_type || '')
    if (p.message_id && p.message_id !== message.message_id) continue
    const seq = event.event_seq || event.id
    if (seq && seen.has(seq)) continue
    if (seq) seen.add(seq)
    if (terminal) continue
    if (paused && kind !== 'run_resumed') continue
    if (kind === 'run_resumed') paused = false
    const id = `${event.run_id}:${String(p.text_id || '')}`
    if (kind.startsWith('agent_text_') && p.text_id) {
      const block = blocks.get(id) || { id, kind: 'text', text: '', role: 'pending', status: 'streaming' }
      latestText = id
      if (kind === 'agent_text_delta') block.text += String(p.text || '')
      if (kind === 'agent_text_completed') {
        block.text = String(p.text ?? block.text)
        block.role = String(p.role || 'pending')
        block.status = block.role === 'interrupted' ? 'interrupted' : 'completed'
      }
      blocks.set(id, block)
    }
    if ((kind === 'progress_delta' || kind === 'progress_completed') && p.block_id) {
      const key = `${event.run_id}:progress:${p.block_id}`
      const previous = blocks.get(key)
      blocks.set(key, { id: key, kind: 'text', role: 'progress', status: kind === 'progress_completed' ? 'completed' : 'streaming',
        text: kind === 'progress_delta' ? (previous?.text || '') + String(p.text || '') : String(p.text || previous?.text || '') })
    }
    if (kind.startsWith('node_') && p.step_id) {
      const key = `${event.run_id}:step:${p.step_id}`
      const step: WorkBlock = { id: key, kind: 'tool', text: String(p.display_name || event.node_name), role: '', status: String(p.status || 'running') }
      steps.set(key, step)
      if (kind === 'node_started') lastStep = key
      if (!concreteToolSteps.has(key) && ['document_read', 'deep_research', 'capability', 'tool_runtime'].includes(event.node_name || '')) blocks.set(key, step)
    }
    if (['tool_call_started', 'tool_call_completed', 'tool_call_failed'].includes(kind)) {
      const key = `${event.run_id}:tool:${p.tool_call_id || p.toolCallId || p.tool_call_record_id || p.id || p.tool_name || p.toolName}`
      // A capability lifecycle and its concrete tool refer to the same activity.
      if (kind === 'tool_call_started') {
        blocks.delete(lastStep)
        concreteToolSteps.add(lastStep)
      }
      blocks.set(key, { id: key, kind: 'tool', text: String(p.tool_name || p.toolName || event.node_name || '工具'), role: '',
        status: kind === 'tool_call_started' ? 'running' : kind === 'tool_call_failed' ? 'failed' : 'completed' })
    }
    if (kind === 'answer_completed' || kind === 'run_completed') {
      const text = String(p.answer ?? '')
      const latest = blocks.get(latestText)
      if (text && latest && latest.role !== 'progress') { latest.text = text; latest.role = 'final'; latest.status = 'completed' }
    }
    if (kind === 'run_paused') paused = true
    if (['run_completed', 'run_failed', 'run_interrupted'].includes(kind)) terminal = true
  }
  const ended = terminal || ['completed', 'failed', 'interrupted', 'waiting_approval'].includes(String(message.status))
  const latest = blocks.get(latestText)
  // Database final snapshots include confirmation/save suffixes; preserve earlier progress.
  if (ended && answer) {
    if (latest && latest.role !== 'progress') {
      latest.text = answer
      if (message.status === 'completed') latest.role = 'final'
      if (message.status === 'interrupted') latest.role = 'interrupted'
    }
    else if (![...blocks.values()].some(b => b.kind === 'text' && b.text === answer)) {
      blocks.set(`${message.run_id}:answer`, { id: `${message.run_id}:answer`, kind: 'text', text: answer, role: 'final', status: 'completed' })
    }
  }
  if (ended || paused) {
    for (const block of [...blocks.values(), ...steps.values()]) {
      if (['running', 'streaming'].includes(block.status)) block.status = paused || message.status === 'waiting_approval' ? 'waiting_approval' : message.status === 'interrupted' ? 'cancelled' : message.status === 'failed' ? 'failed' : 'ended'
    }
  }
  return { blocks: [...blocks.values()].filter(b => b.kind === 'tool' || b.text.trim()), steps: [...steps.values()], active: !ended && !paused }
}
