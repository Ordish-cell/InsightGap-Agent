import type { AgentChatMessage, AgentEvent } from '../../api/types'

/** Shared live/replay projection. Never apply an old run to another bubble. */
export function projectChatEvent(message: AgentChatMessage, event: AgentEvent): AgentChatMessage {
  if (message.role !== 'assistant' || Number(message.run_id) !== Number(event.run_id)) return message
  const payload = (event.payload || {}) as Record<string, unknown>
  if (payload.message_id && payload.message_id !== message.message_id) return message
  if (['interrupted', 'failed'].includes(String(message.status))) return message
  const seq = Number(event.event_seq || event.id || 0)
  if (seq && seq <= Number(message.metadata?.chat_event_seq || 0)) return message
  if (seq) message = { ...message, metadata: { ...message.metadata, chat_event_seq: seq } }
  switch (event.event_type) {
    case 'agent_text_started':
      return { ...message, content: '', metadata: { ...message.metadata, native_text_id: payload.text_id }, status: 'streaming' }
    case 'agent_text_delta':
      if (message.metadata?.native_text_id !== payload.text_id) return message
      return { ...message, content: (message.content || '') + String(payload.text || ''), status: 'streaming' }
    case 'agent_text_completed':
      if (message.metadata?.native_text_id !== payload.text_id) return message
      return { ...message, content: payload.role === 'progress' ? '' : String(payload.text || ''),
        metadata: { ...message.metadata, native_text_role: payload.role } }
    case 'answer_delta':
      if (payload.text_id && payload.text_id === message.metadata?.native_text_id) return message
      if (['interrupted', 'failed', 'completed'].includes(String(message.status))) return message
      return { ...message, status: 'streaming', content: (message.content || '') + String(payload.text || '') }
    case 'answer_completed':
      return { ...message, content: String(payload.answer ?? message.content ?? '') }
    case 'run_interrupted':
      return { ...message, status: 'interrupted', content: String(payload.answer ?? message.content ?? ''), error_message: String(payload.error || '') }
    case 'run_failed':
      return { ...message, status: 'failed', error_message: String(payload.error || ''), content: String(payload.answer || message.content || '本次回复未完成，请稍后重试。') }
    case 'run_completed':
      return { ...message, status: 'completed', content: String(payload.answer ?? message.content ?? '') }
    case 'run_paused':
    case 'approval_required':
      return { ...message, status: 'waiting_approval' }
    default:
      return message
  }
}

export function restoreChatMessage(message: AgentChatMessage, events: AgentEvent[]): AgentChatMessage {
  if (message.role !== 'assistant') return message
  // Final database snapshots already contain all text. Replay is only needed
  // for an unfinished snapshot (the service persists its body at completion).
  if (['completed', 'interrupted', 'failed', 'waiting_approval'].includes(String(message.status))) return message
  let restored = { ...message, content: '', metadata: { ...message.metadata, chat_event_seq: 0 } }
  const seen = new Set<number>()
  for (const event of events) {
    const seq = Number(event.event_seq || event.id || 0)
    if (!seq || seen.has(seq)) continue
    seen.add(seq)
    restored = projectChatEvent(restored, event) as typeof restored
  }
  return restored
}
