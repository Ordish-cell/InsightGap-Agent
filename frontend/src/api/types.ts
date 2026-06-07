export type UnknownRecord = Record<string, unknown>

export interface ApiEnvelope<T> {
  success?: boolean
  data?: T
  error?: { code?: string; message?: string; details?: UnknownRecord }
  request_id?: string
}

export interface ApiListResponse<T> {
  items?: T[]
  results?: T[]
  data?: T[]
  total?: number
  [key: string]: unknown
}

export interface CurrentUser {
  id: number
  email: string
  nickname?: string
  status?: string
  [key: string]: unknown
}

export interface AuthResponse {
  access_token: string
  token_type?: string
  [key: string]: unknown
}

export interface EvidenceItem {
  title?: string
  source_url?: string
  url?: string
  document_id?: number | string
  chunk_id?: number | string
  score?: number
  snippet?: string
  quote?: string
  summary?: string
  metadata?: UnknownRecord
  [key: string]: unknown
}

export interface FeedScore {
  personal_relevance?: number
  novelty?: number
  cross_domain_distance?: number
  opportunity_value?: number
  source_credibility?: number
  actionability?: number
  profile_match?: number
  semantic_memory_match?: number
  [key: string]: unknown
}

export interface FeedCard {
  id: number
  title: string
  display_title?: string
  original_title?: string
  one_sentence_value?: string
  why_you?: string
  why_relevant?: string
  benefit?: string
  information_gap?: string
  next_action?: string
  summary?: string
  exposure_bucket?: string
  relation_type?: string
  source_type?: string
  domain?: string
  source_url?: string
  final_score?: number
  score?: FeedScore
  score_detail?: FeedScore & { original_title?: string; why_relevant?: string; benefit?: string; next_action?: string; summary?: string; [key: string]: unknown }
  evidence?: EvidenceItem[]
  suggested_actions?: string[]
  low_confidence?: boolean
  created_at?: string
  [key: string]: unknown
}

export interface ResearchRun {
  id: string
  query?: string
  status?: string
  summary?: string
  markdown_report?: string
  evidence?: EvidenceItem[]
  findings?: UnknownRecord[]
  risks?: UnknownRecord[]
  opportunities?: UnknownRecord[]
  suggested_actions?: UnknownRecord[]
  artifact_id?: number
  skill_draft_id?: number
  agent_run_id?: number
  feed_card_id?: number
  created_at?: string
  completed_at?: string
  [key: string]: unknown
}

export interface AgentRun {
  run_id?: number
  id?: number
  conversation_id?: string
  thread_id?: string
  route?: string
  status?: string
  elapsed_ms?: number
  answer?: string
  final_answer?: string
  final_output?: string
  final_response?: UnknownRecord
  final_payload?: UnknownRecord
  visible_thoughts?: unknown[]
  thinking_summary?: string[]
  assistant_message?: AgentChatMessage
  user_message?: AgentChatMessage
  conversation?: AgentConversation
  langgraphstatus?: UnknownRecord
  artifacts?: Artifact[]
  matched_skill?: SkillDraft & { match_score?: number; match_reason?: string; auto_use?: boolean }
  candidate_skills?: Array<SkillDraft & { match_score?: number; match_reason?: string; auto_use?: boolean }>
  created_skill_draft?: SkillDraft
  reusable_score?: number
  tool_call?: McpToolCall
  evaluation?: UnknownRecord
  [key: string]: unknown
}

export interface AgentConversation {
  id?: number
  conversation_id: string
  thread_id?: string
  title?: string
  source?: string
  status?: string
  selected_feed_card_id?: number | null
  selected_feed_card_title?: string
  metadata?: UnknownRecord
  last_message_preview?: string
  last_run_id?: number | null
  message_count?: number
  messages?: AgentChatMessage[]
  created_at?: string
  updated_at?: string
  last_active_at?: string
  [key: string]: unknown
}

export interface AgentChatMessage {
  id?: number
  message_id: string
  conversation_id?: string
  run_id?: number | null
  thread_id?: string
  role: 'user' | 'assistant' | 'system'
  content?: string
  status?: string
  elapsed_ms?: number | null
  langgraphstatus?: UnknownRecord
  steps?: AgentRunStep[]
  error_message?: string
  metadata?: UnknownRecord
  created_at?: string
  updated_at?: string
  attachments?: ChatAttachment[]
  [key: string]: unknown
}

export type ChatAttachmentKind = 'image' | 'document' | 'audio' | 'video' | 'file'

export type ChatAttachmentStatus = 'queued' | 'uploading' | 'uploaded' | 'failed'

export interface ChatAttachment {
  document_id: number
  filename: string
  file_type: string
  mime_type?: string
  kind: ChatAttachmentKind
  size?: number
  preview_url?: string
  status?: string
  ingest_status?: string
  error?: string
}

export interface AgentRunStep {
  key?: string
  title?: string
  status?: string
  summary?: string
  detail?: string
  node_name?: string
  started_at?: string
  completed_at?: string | null
  elapsed_ms?: number
  events_count?: number
  [key: string]: unknown
}

export interface AgentStep {
  id: number
  node_name?: string
  status?: string
  input?: UnknownRecord
  output?: UnknownRecord
  [key: string]: unknown
}

export interface AgentEvent {
  id?: number
  run_id?: number
  thread_id?: string
  event_type?: string
  node_name?: string
  payload?: UnknownRecord
  created_at?: string
  [key: string]: unknown
}

export interface Artifact {
  id: number
  title?: string
  artifact_type?: string
  file_path?: string
  public_url?: string
  metadata?: UnknownRecord
  created_at?: string
  content?: string
  [key: string]: unknown
}

export interface MemoryItem {
  id: number
  content?: string
  memory_type?: string
  importance?: number
  metadata?: UnknownRecord
  created_at?: string
  [key: string]: unknown
}

export interface MemorySummary {
  counts?: Array<{ memory_type?: string; count?: number; avg_importance?: number }>
  recent?: MemoryItem[]
  [key: string]: unknown
}

export interface SkillDraft {
  id: number
  name?: string
  description?: string
  trigger_text?: string
  input_schema?: UnknownRecord
  context_recipe?: unknown[]
  tool_plan?: unknown[]
  output_schema?: UnknownRecord
  safety_level?: string
  eval_checks?: unknown[]
  status?: string
  version?: number
  [key: string]: unknown
}

export interface ApprovalItem {
  id: number
  run_id?: number
  approval_type?: string
  title?: string
  description?: string
  payload?: UnknownRecord
  status?: string
  created_at?: string
  [key: string]: unknown
}

export interface McpTool {
  id?: number
  server_id?: number
  name: string
  description?: string
  category?: string
  input_schema?: UnknownRecord
  output_schema?: UnknownRecord
  safety_level?: string
  enabled?: boolean
  requires_approval?: boolean
  metadata?: UnknownRecord
  [key: string]: unknown
}

export interface McpToolCall {
  id?: number
  user_id?: number
  agent_run_id?: number
  tool_id?: number
  tool_name?: string
  safety_level?: string
  status?: string
  input?: UnknownRecord
  output?: UnknownRecord
  error?: string
  approval_id?: number
  created_at?: string
  completed_at?: string
  [key: string]: unknown
}

export interface HealthResponse {
  status?: string
  mysql?: UnknownRecord
  redis?: UnknownRecord
  qdrant?: UnknownRecord
  feed_sources?: unknown
  open_deep_research?: UnknownRecord
  agent_runtime?: UnknownRecord
  mcp?: UnknownRecord
  [key: string]: unknown
}
