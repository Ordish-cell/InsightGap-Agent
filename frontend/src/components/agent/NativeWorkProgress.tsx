import type { AgentChatMessage, AgentEvent } from '../../api/types'
import { MarkdownRenderer } from '../common/MarkdownRenderer'
import { ApprovalCard } from './ApprovalCard'
import { approvalFrom } from './AgentThoughtStream'
import { projectWorkProgress } from './workProgress'

export function NativeWorkProgress({ message, answer, locale, onApprove, onReject }: {
  message: AgentChatMessage & { trace_events?: AgentEvent[] }; answer: string; locale: 'zh' | 'en'
  onApprove: (id: number) => void; onReject: (id: number) => void
}) {
  const trace = projectWorkProgress(message, message.trace_events || [], answer)
  const { approvalId, cardData } = approvalFrom(message, locale)
  const labels: Record<string, string> = locale === 'zh'
    ? { running: '进行中', completed: '完成', failed: '失败', cancelled: '已停止', ended: '已结束', waiting_approval: '等待确认' }
    : { running: 'Running', completed: 'Completed', failed: 'Failed', cancelled: 'Stopped', ended: 'Ended', waiting_approval: 'Awaiting approval' }
  return <div className="native-work-progress">
    {trace.blocks.map(block => block.kind === 'text'
      ? <div key={block.id} data-work-block={block.id} data-text-role={block.role} className="work-text-block answer-content"><MarkdownRenderer content={block.text} /></div>
      : <div key={block.id} className={`work-tool-row ${block.status}`}><span aria-hidden="true">{block.status === 'running' ? '◦' : block.status === 'failed' ? '!' : '·'}</span><span>{block.text}</span><small>{labels[block.status] || block.status}</small></div>)}
    {trace.active ? <div className="work-live-status" role="status"><span className="thinking-dot active" />{locale === 'zh' ? '正在处理…' : 'Working…'}</div> : null}
    {trace.steps.length ? <details className="live-progress-details"><summary>{locale === 'zh' ? '执行详情' : 'Execution details'}</summary><ul>{trace.steps.map(step => <li key={step.id}>{step.text} · {labels[step.status] || step.status}</li>)}</ul></details> : null}
    {approvalId ? <ApprovalCard data={cardData} locale={locale} onApprove={onApprove} onReject={onReject} /> : null}
  </div>
}
