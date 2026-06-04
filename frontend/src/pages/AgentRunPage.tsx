import { AgentChatPanel } from '../components/agent/AgentChatPanel'
import { PageHeader } from '../components/common/PageHeader'

export function AgentRunPage() {
  return (
    <section className="workbench-page agent-debug-page">
      <PageHeader title="Agent debug" description="Compatibility entry for direct Agent Runtime runs, streaming events, and step inspection." />
      <AgentChatPanel source="agent_page" pageContext={{ page: 'agent' }} placeholder="Enter a read-only or local-write Agent task" initialTitle="Run an Agent task" debug />
    </section>
  )
}
