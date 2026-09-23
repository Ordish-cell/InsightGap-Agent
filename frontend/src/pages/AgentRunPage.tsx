import { AgentChatPanel } from '../components/agent/AgentChatPanel'
import { PageHeader } from '../components/common/PageHeader'

export function AgentRunPage() {
  return (
    <section className="workbench-page agent-debug-page">
      <PageHeader title="Agent 调试" description="查看运行事件、步骤与工具执行详情。" />
      <AgentChatPanel source="agent_page" pageContext={{ page: 'agent' }} placeholder="输入需要调试的任务…" initialTitle="运行一个任务" locale="zh" debug />
    </section>
  )
}
