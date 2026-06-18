# Checkpoint Architecture

## Overview

The agent runtime uses LangGraph interrupt-based checkpointing with PostgresSaver
as the production backend.  When a tool requires user approval (L3/L4 risk level),
the graph pauses via `langgraph.types.interrupt()`, the checkpointer saves state
to PostgreSQL, and the graph can be resumed from a different process via
`Command(resume=...)`.

## Old vs New Architecture

### Old: END-based (deprecated)

```
User → graph.invoke() → tool_agent → status="waiting_approval"
                                    → dispatch returns END
                                    → graph terminates
                                    → agent_service saves full graph_state
                                      to agent_runs.graph_state (JSONB)
                                    → SSE: approval_required + run_paused

User approves → agent_service loads graph_state from DB
              → graph.invoke() from entry_point (full replay)
              → tool_agent detects resume context in graph_state
              → continues execution
```

Problems with END-based approach:
- **Full graph replay**: Every node re-executes from entry_point, wasting LLM calls
- **State staleness**: `approval_required`, `route`, `error` fields leak across replays,
  requiring aggressive `sanitize_resume_final_state()` cleanup
- **Large JSONB storage**: Full graph_state serialized to `agent_runs.graph_state`
  on every pause
- **No checkpoint integrity**: If the graph crashes mid-node, partial work is lost

### New: interrupt + Command(resume) + PostgresSaver

```
User → graph.ainvoke() → tool_agent → interrupt({"type": "approval_required", ...})
                                     → LangGraph saves checkpoint to PostgresSaver
                                       (4 tables: checkpoints, checkpoint_blobs,
                                        checkpoint_writes, checkpoint_migrations)
                                     → GraphInterrupt raised
                                     → agent_service catches it, saves minimal
                                       pause state to agent_runs.graph_state
                                     → SSE: approval_required + run_paused

User approves → tool_executor.execute_approved_tool() (OUTSIDE the graph)
              → agent_service calls runtime.resume_from_interrupt(
                  Command(resume={"action": "approved", "tool_result": ...}),
                  thread_id="run:{run_id}"
                )
              → LangGraph loads checkpoint from PostgresSaver
              → graph continues from interrupt() call site
              → NO graph replay, NO stale state cleanup needed
```

Advantages:
- **No graph replay**: Resumes exactly at the interrupt point
- **Clean state**: No stale approval fields to clean up
- **Cross-process**: Different worker can resume the graph (checkpoint in PostgreSQL)
- **Durable**: Survives process restart, deployment, scaling events
- **Smaller DB footprint**: Only checkpoint data in PostgreSQL; agent_runs.graph_state
  stores minimal metadata

## Key Design Decisions

### thread_id = `run:{run_id}`

Each run gets a unique checkpoint thread_id: `f"run:{run.id}"`.  This is NOT
the same as the conversation thread_id (`user:{user_id}:conversation:{conversation_id}`).
One per run — each approval pause is a separate checkpoint.

### Tool execution outside the graph

Tools are executed in `agent_service.py`, NOT inside the LangGraph node.  The
resume flow is:

1. `tool_executor.execute_approved_tool()` — runs the actual tool
2. `Command(resume={"action": "approved", "tool_result": result})` — passes result
   back to the graph
3. `tool_agent._handle_tool_resume_approved()` — accepts the pre-executed result

This keeps tool execution outside the checkpoint/serialization boundary.

### _stream_queue removed from state

`_stream_queue` (asyncio.Queue) is not JSON-serializable.  It's stored on
`AgentRuntime._stream_queue` and `RuntimeNodes._stream_queue`, never in
`AgentRuntimeState`.  Before invoke: `state.pop("_stream_queue", None)`.

### approval_pause_mode

- `"interrupt"` → new path: `Command(resume=...)` from checkpoint
- `"end"` → legacy path: full graph replay from entry_point (deprecated)

New runs always set `"interrupt"`.  The `"end"` value only exists for runs
created before Phase 8 deployment.

## Configuration

### Production (.env)

```bash
AGENT_LANGGRAPH_CHECKPOINTER_ENABLED=true
AGENT_CHECKPOINTER_BACKEND=postgres
AGENT_CHECKPOINTER_REQUIRE_DURABLE=true
AGENT_APPROVAL_INTERRUPT_ENABLED=true
# Optional: separate DB for checkpoints
# AGENT_CHECKPOINTER_DATABASE_URL=postgresql://user:pass@host:5432/checkpoint_db
```

### Dev/Test (.env)

```bash
AGENT_LANGGRAPH_CHECKPOINTER_ENABLED=true
AGENT_CHECKPOINTER_BACKEND=postgres
AGENT_CHECKPOINTER_REQUIRE_DURABLE=false   # won't fail if PG is down
AGENT_APPROVAL_INTERRUPT_ENABLED=true
```

### Dev without PostgreSQL

```bash
AGENT_LANGGRAPH_CHECKPOINTER_ENABLED=true
AGENT_CHECKPOINTER_BACKEND=memory          # InMemorySaver — lost on restart
AGENT_CHECKPOINTER_REQUIRE_DURABLE=false
AGENT_APPROVAL_INTERRUPT_ENABLED=true
```

## PostgreSQL Checkpoint Tables

| Table | Purpose |
|---|---|
| `checkpoints` | Main checkpoint state (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint JSONB) |
| `checkpoint_blobs` | Channel values (thread_id, channel, version, blob) |
| `checkpoint_writes` | Pending writes (thread_id, checkpoint_id, task_id, channel, value) |
| `checkpoint_migrations` | Schema migration tracking |

All tables use `thread_id` as part of their primary key.

## Checkpoint Cleanup

Finished runs' checkpoints can be cleaned up to save storage:

```python
from src.web_app.agent.runtime.checkpoint_cleanup import cleanup_checkpoints

# Dry run — see what would be deleted
summary = cleanup_checkpoints(dry_run=True)

# Real cleanup with default TTLs (completed=7d, failed=30d, cancelled=7d)
summary = cleanup_checkpoints()

# Custom TTLs
summary = cleanup_checkpoints(ttl_days={"completed": 3, "failed": 14})
```

**Safety guarantees:**
- `waiting_approval` / `paused` / `resuming` / `running` runs are NEVER cleaned
- Each eligible run is verified against the agent_runs table before deletion
- Deletion is by `thread_id` (= `run:{run_id}`) across all 3 data tables
- Dry-run mode shows exactly what would be deleted without making changes

## Troubleshooting

### "FATAL: backend=postgres requires a connection string"

The checkpointer is configured for postgres but no database URL was provided.
Set `AGENT_CHECKPOINTER_DATABASE_URL` or ensure the main `DATABASE_URL` is
configured.

### "FATAL: missing checkpoint tables: [checkpoints, ...]"

The PostgresSaver tables don't exist.  Run:

```python
from langgraph.checkpoint.postgres import PostgresSaver
ctx = PostgresSaver.from_conn_string(conn_string)
saver = ctx.__enter__()
saver.setup()
```

Or set `AGENT_CHECKPOINTER_REQUIRE_DURABLE=false` temporarily.

### "resume_from_interrupt requires a thread_id"

The thread_id wasn't set when the graph was first invoked.  Ensure
`state["thread_id"] = f"run:{run_id}"` is set before `graph.ainvoke()`.

### "Command resume can't find checkpoint"

Possible causes:
- `thread_id` mismatch between pause and resume (must be identical)
- Checkpointer backend changed between pause and resume (e.g., postgres → memory)
- Checkpoint was cleaned up by the cleanup job
- PostgreSQL connection issue — check if the database is reachable

### "MemorySaver fallback" warning in production

This means PostgresSaver failed to initialize (bad connection string, DB down)
and the system fell back to InMemorySaver.  Check the logs for the specific
error.  With `AGENT_CHECKPOINTER_REQUIRE_DURABLE=true`, this fallback is blocked
and the app will fail fast at startup.

### "LEGACY_APPROVAL_RESUME" log warning

An old run (created before Phase 8) is being resumed via the END-based path.
This is normal for runs that were paused before the migration.  Once all old
waiting_approval runs are resolved, the legacy path can be deleted.
