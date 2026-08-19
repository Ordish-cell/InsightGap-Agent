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

export interface ResearchRunMetadata {
  source?: string
  engine?: string
  used_fallback?: boolean | null
  odr_enabled?: boolean
  odr_error?: string | null
  odr_error_type?: string | null
  adapter?: string
  depth?: string
  feed_card_id?: number | null
  card_snapshot?: UnknownRecord
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
  sources?: UnknownRecord[]
  artifact_id?: number
  skill_draft_id?: number
  agent_run_id?: number
  feed_card_id?: number
  error?: string
  error_message?: string
  metadata?: ResearchRunMetadata
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
  pipeline_steps?: unknown[]
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

export type LlmProtocol = 'openai_chat_completions' | 'openai_responses' | 'anthropic_messages' | 'google_generate_content' | 'ollama_chat'

export interface LlmProviderField {
  key: string
  label: string
  kind: 'text' | 'secret' | 'secret_json' | 'select' | 'url' | 'json'
  required: boolean
  default?: unknown
  options?: Array<{ label: string; value: string }>
  placeholder?: string
}

export interface LlmPresetModel {
  model_id: string
  display_name: string
  tier: 'strong' | 'balanced' | 'fast'
  capabilities?: Record<string, boolean>
}

export interface LlmProviderDefinition {
  key: string
  label: string
  protocol: LlmProtocol
  protocols: LlmProtocol[]
  fields: LlmProviderField[]
  models: LlmPresetModel[]
  discovery: string
  capabilities: Record<string, boolean>
}

export interface LlmModelConfig {
  id: number
  connection_id: number
  model_id: string
  display_name: string
  source: 'preset' | 'discovered' | 'manual'
  capabilities: Record<string, boolean>
  enabled: boolean
}

export interface LlmConnection {
  id: number
  provider: string
  protocol: LlmProtocol
  display_name: string
  fields: Record<string, unknown>
  secrets: Record<string, { configured: boolean; masked: string }>
  revision: number
  status: 'draft' | 'active' | 'deleted'
  last_test_status: 'untested' | 'passed' | 'failed'
  last_test_error?: string
  last_tested_at?: string | null
  models: LlmModelConfig[]
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
  event_seq?: number
  schema_version?: number
  run_id?: number
  thread_id?: string
  event_type?: string
  node_name?: string
  visibility?: 'user' | 'trace' | 'internal' | string
  display_channel?: 'thinking' | 'answer' | 'tool' | 'status' | string
  payload?: UnknownRecord
  created_at?: string
  [key: string]: unknown
}

export interface AgentReplayPage {
  run_id: number
  events: AgentEvent[]
  after_seq: number
  next_seq: number
  until_seq: number
  has_more: boolean
}

export type AgentTraceEventType =
  | 'visible_thought_delta'
  | 'visible_progress_delta'
  | 'tool_call_started'
  | 'tool_call_completed'
  | 'tool_call_failed'
  | 'approval_required'
  | 'approval_granted'
  | 'approval_rejected'
  | 'answer_started'
  | 'answer_delta'
  | 'answer_completed'
  | 'run_completed'
  | 'run_failed'

export interface AgentToolTrace {
  id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'waiting_approval' | 'rejected' | string
  argsPreview?: unknown
  outputPreview?: string
  error?: string
  recordId?: string | number
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

export type LongTermMemoryType = 'semantic' | 'episodic'
export type LongTermMemoryStatus = 'active' | 'archived' | 'low_confidence' | 'superseded'

export interface LongTermMemoryItem {
  id: number; memory_type: LongTermMemoryType; content: string
  category?: string; importance: number; effective_importance?: number
  confidence?: number; status: LongTermMemoryStatus; stability?: string
  evidence_count?: number; last_seen_at?: string; created_at?: string
  updated_at?: string; metadata?: UnknownRecord
  [key: string]: unknown
}

export interface LongTermMemoryListResponse {
  items: LongTermMemoryItem[]; total: number; page: number; page_size: number
}
