import type { AgentStep, FeedCard, ResearchRun } from './types'

export function normalizeFeedCard(raw: unknown): FeedCard {
  const item = raw as FeedCard & Record<string, unknown>
  const detail = (item.score_detail || {}) as Record<string, unknown>

  // display_title takes priority over title for the main display field.
  // Falls back through: display_title -> title_zh -> chinese_title ->
  //   score_detail.display_title -> score_detail.title_zh -> title
  const displayTitle = String(
    item.display_title ||
    item.title_zh ||
    (item as Record<string, unknown>).chinese_title ||
    detail.display_title ||
    detail.title_zh ||
    item.title ||
    '未命名信息差'
  )

  // original_title preserves the English original
  const originalTitle = String(
    item.original_title ||
    detail.original_title ||
    ''
  )

  return {
    ...item,
    id: Number(item.id),
    title: displayTitle,
    display_title: displayTitle,
    original_title: originalTitle,
    one_sentence_value: String(item.one_sentence_value || item.summary || ''),
    why_you: String(item.why_you || ''),
    why_relevant: String(item.why_relevant || detail.why_relevant || item.why_you || ''),
    benefit: String(item.benefit || detail.benefit || ''),
    information_gap: String(item.information_gap || ''),
    next_action: String(item.next_action || detail.next_action || ''),
    summary: String(item.summary || detail.summary || ''),
    exposure_bucket: String(item.exposure_bucket || item.relation_type || ''),
    relation_type: String(item.relation_type || item.exposure_bucket || ''),
    source_type: String(item.source_type || ''),
    domain: String(item.domain || item.source_domain || ''),
    final_score: Number(item.final_score || 0),
    evidence: Array.isArray(item.evidence) ? item.evidence : [],
    suggested_actions: Array.isArray(item.suggested_actions) ? item.suggested_actions as string[] : [],
  }
}

export function normalizeResearchRun(raw: unknown): ResearchRun {
  const item = raw as ResearchRun
  return {
    ...item,
    id: String(item.id || ''),
    status: item.status || 'pending',
    evidence: Array.isArray(item.evidence) ? item.evidence : [],
    findings: Array.isArray(item.findings) ? item.findings : [],
  }
}

export function normalizeAgentStep(raw: unknown): AgentStep {
  const item = raw as AgentStep
  return {
    ...item,
    id: Number(item.id),
    node_name: item.node_name || 'step',
    status: item.status || 'completed',
    input: item.input || {},
    output: item.output || {},
  }
}
