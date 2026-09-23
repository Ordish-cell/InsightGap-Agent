const statusMap: Record<string, string> = {
  active: '生效中',
  archived: '已归档',
  superseded: '已替代',
  low_confidence: '低置信',
  semantic: '长期设定',
  episodic: '行为事件',
  preferred_name: '称呼',
  response_language: '回答语言',
  script_preference: '简繁偏好',
  tech_stack: '技术栈',
  preference: '产品偏好',
  project_goal: '项目目标',
  boundary: '当前边界',
  workflow_pattern: '任务模式',
  product_principle: '产品原则',
  uncategorized: '其他',
  markdown_report: '研究报告',
  markdown: 'Markdown',
  text: '文本',
  interrupted: '已停止',
  cancelled: '已取消',
  expired: '已过期',
  completed: '已完成',
  pending: '等待中',
  running: '运行中',
  failed: '失败',
  draft: '草稿',
  approved: '已批准',
  disabled: '已停用',
  rejected: '已拒绝',
  waiting_approval: '等待审批',
  blocked: '已阻断',
  new: '新卡片',
}

const relationMap: Record<string, string> = {
  explicit_related: '显性相关',
  adjacent_domain: '邻近领域',
  far_domain: '远域启发',
}

const safetyMap: Record<string, string> = {
  L0_READ_ONLY: 'L0 只读',
  L1_DRAFT: 'L1 草稿',
  L2_LOCAL_WRITE: 'L2 本地写入',
  L3_EXTERNAL_WRITE: 'L3 需审批',
  L4_HIGH_RISK: 'L4 默认阻断',
  read_only: '只读',
}

const sourceMap: Record<string, string> = {
  arxiv: 'Arxiv',
  github: 'GitHub',
  rss: 'RSS',
  web: '网页',
  manual: '内置种子',
}

export function statusLabel(value?: string): string {
  return statusMap[value || ''] || value || '未知'
}

export function relationLabel(value?: string): string {
  return relationMap[value || ''] || value || '未分类'
}

export function safetyLevelLabel(value?: string): string {
  return safetyMap[value || ''] || value || '未标记'
}

export function sourceTypeLabel(value?: string): string {
  return sourceMap[value || ''] || value || '来源'
}

export function shortId(value?: string | number): string {
  const text = String(value || '')
  return text.length > 12 ? `${text.slice(0, 8)}...` : text
}
