import { FormEvent, useState } from 'react'

import * as agent from '../api/agent'
import type { AgentRun, AgentStep } from '../api/types'
import { ErrorState } from '../components/common/ErrorState'
import { JsonBlock } from '../components/common/JsonBlock'
import { PageHeader } from '../components/common/PageHeader'
import { StatusPill } from '../components/common/StatusPill'

export function AgentRunPage() {
  const [userInput, setUserInput] = useState('')
  const [run, setRun] = useState<AgentRun | null>(null)
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [stream, setStream] = useState<string[]>([])
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    setStream([])
    try {
      const nextRun = await agent.createRun({ user_input: userInput })
      setRun(nextRun)
      const runId = nextRun.run_id || nextRun.id
      if (runId) {
        const stepResult = await agent.getSteps(runId)
        setSteps(stepResult.steps || [])
        const source = agent.createRunStream(runId, {
          onMessage: (message) => setStream((items) => [...items.slice(-12), message.data]),
          onError: () => setStream((items) => [...items, 'stream unavailable']),
        })
        setTimeout(() => source.close(), 2500)
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Agent run failed')
    }
  }

  return (
    <>
      <PageHeader title="Agent" description="Route a task through LangGraph Runtime, with approvals, MCP audits, RAG, Research, Artifacts, Memory, and Skills." />
      <div className="panel">
        <p className="muted">Safety: external writes create drafts or approvals. L3 requires approval. L4 is blocked by default.</p>
        <form className="stack" onSubmit={submit}>
          <label>Task<textarea className="textarea" value={userInput} onChange={(event) => setUserInput(event.target.value)} required /></label>
          <button className="button" style={{ width: 'fit-content' }}>Run Agent</button>
        </form>
      </div>
      {error ? <ErrorState message={error} /> : null}
      {run ? (
        <div className="split">
          <div className="stack">
            <div className="panel stack"><div className="row"><StatusPill value={run.status} /><StatusPill value={run.route} /></div><p>{run.final_output}</p><JsonBlock value={run.tool_call || run.evaluation} /></div>
            <div className="panel"><h2>Steps</h2>{steps.map((step) => <div className="evidence-item" key={step.id}><div><strong>{step.node_name}</strong><p>{step.status}</p></div><JsonBlock value={{ input: step.input, output: step.output }} /></div>)}</div>
          </div>
          <aside className="panel"><h2>SSE stream</h2>{stream.length ? <JsonBlock value={stream} /> : <p className="muted">No stream events yet.</p>}</aside>
        </div>
      ) : null}
    </>
  )
}
