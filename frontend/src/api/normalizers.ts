import type { AgentStep, FeedCard, ResearchRun } from './types'

export function normalizeFeedCard(raw: unknown): FeedCard {
  const item = raw as FeedCard & Record<string, unknown>
  return {
    ...item,
    id: Number(item.id),
    title: String(item.title || '未命名信息差'),
    one_sentence_value: String(item.one_sentence_value || item.summary || ''),
    why_you: String(item.why_you || ''),
    information_gap: String(item.information_gap || ''),
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
