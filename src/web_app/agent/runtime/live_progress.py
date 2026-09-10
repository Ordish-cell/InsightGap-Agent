"""Real node lifetimes and evidence-backed results, independent of UI wording."""
import asyncio
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

current_step = ContextVar("current_live_step", default=None)
step_started_at = ContextVar("live_step_started_at", default=None)
NAMES = {"chat_entry": "判断处理方式", "permission_guard": "检查操作权限",
         "home_intent_react": "判断任务需要", "planner": "安排执行步骤",
         "parallel_prefetch": "获取相关上下文", "parallel_read_stage": "准备上下文",
         "context_builder": "读取会话上下文", "skill_matcher": "匹配可用能力",
         "supervisor_observer": "检查执行计划", "llm_supervisor_route": "选择执行路径",
         "research_agent": "研究问题", "rag_agent": "检索资料", "tool_agent": "执行工具",
         "artifact_agent": "生成文件", "memory_agent": "更新记忆", "skill_agent": "整理技能",
         "evaluator": "检查结果", "final_response": "生成回复"}


def publish_findings(db, state, name, step_id):
    from src.web_app.agent.runtime.event_ledger import publish_event
    result = state.get({"research_agent": "research_result", "rag_agent": "rag_result",
                        "tool_agent": "tool_result"}.get(name, ""))
    if not isinstance(result, dict):
        return
    output = result.get("output") if isinstance(result.get("output"), dict) else result
    # Only explicit public results, never reasoning/thought fields or raw tool args.
    text = output.get("summary") or output.get("answer")
    if name == "rag_agent":
        evidence = [item for item in output.get("evidence", []) if isinstance(item, dict)]
        if not evidence:
            return
        names = list(dict.fromkeys(str(item.get("source_title") or item.get("source_name") or "") for item in evidence))
        sources = "、".join(name for name in names[:3] if name)
        text = f"从{sources or '资料'}中检索到 {len(evidence)} 段相关内容。"
    if not text and isinstance(output.get("results"), list) and output["results"]:
        titles = [str(item.get("title") or item.get("url") or "") for item in output["results"] if isinstance(item, dict)]
        text = f"检索到 {len(output['results'])} 条结果：" + "；".join(titles[:3])
    if not isinstance(text, str) or not text.strip() or text.startswith("[document_qa_context]"):
        return
    refs = []
    if name == "rag_agent":
        refs = [{key: item[key] for key in ("document_id", "chunk_id", "source_title", "source_url") if key in item} for item in evidence]
    for item in output.get("sources", output.get("results", [])) or []:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            refs.append({"url": item["url"], "title": str(item.get("title") or item["url"])})
    publish_event(db, None, state["run_id"], "progress_completed", {
        "run_id": state["run_id"], "step_id": step_id, "block_id": f"{step_id}:result",
        "text": text, "references": refs, "status": "completed",
    }, node_name=name, user_id=state.get("user_id"), thread_id=state.get("thread_id"))


async def run_node(name, node, db, state, args, kwargs):
    if db is None or state.get("interaction_version") != 2:
        return await node(state, *args, **kwargs)
    from src.web_app.agent.runtime.event_ledger import publish_event
    step_id = uuid4().hex
    parent = current_step.get()
    marker = current_step.set(step_id)
    timing_marker = step_started_at.set(datetime.now(UTC).replace(tzinfo=None))
    started = perf_counter()
    base = {"step_id": step_id, "parent_step_id": parent, "display_name": NAMES.get(name, name)}
    def emit(kind, **extra):
        publish_event(db, None, state["run_id"], kind, {**base, **extra}, node_name=name,
                      user_id=state.get("user_id"), thread_id=state.get("thread_id"))
    try:
        emit("node_started", status="running")
        try:
            result = await node(state, *args, **kwargs)
            from src.web_app.agent.runtime.chat_control import check_active
            check_active(state["run_id"])
        except asyncio.CancelledError:
            emit("node_cancelled", status="cancelled", elapsed_ms=(perf_counter() - started) * 1000)
            raise
        except BaseException as exc:
            from langgraph.errors import GraphInterrupt
            if isinstance(exc, GraphInterrupt):
                emit("node_paused", status="waiting_approval")
            else:
                emit("node_failed", status="failed", elapsed_ms=(perf_counter() - started) * 1000)
            raise
        status = result.get("status")
        event_type = "node_paused" if status == "waiting_approval" else "node_failed" if status == "failed" else "node_completed"
        emit(event_type, status=status if status in {"waiting_approval", "failed"} else "completed", elapsed_ms=(perf_counter() - started) * 1000)
        if event_type == "node_completed":
            publish_findings(db, result, name, step_id)
        return result
    finally:
        step_started_at.reset(timing_marker)
        current_step.reset(marker)
