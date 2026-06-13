from __future__ import annotations

import asyncio
import copy
import time
from typing import Any

from src.web_app.agent.runtime.state import AgentRuntimeState
from src.web_app.db.session import SessionLocal
from src.web_app.services.rag_service import rag_service

RAG_PREPARE_TIMEOUT_SECONDS = 4.0
CONTEXT_SKILL_TIMEOUT_SECONDS = 8.0
OVERALL_TIMEOUT_SECONDS = 9.0


async def parallel_read_stage(
    state: AgentRuntimeState,
    nodes: Any,
    payload: dict[str, Any] | None = None,
    *,
    rag_prepare_timeout: float = RAG_PREPARE_TIMEOUT_SECONDS,
    context_skill_timeout: float = CONTEXT_SKILL_TIMEOUT_SECONDS,
    overall_timeout: float = OVERALL_TIMEOUT_SECONDS,
) -> AgentRuntimeState:
    """Prepare read-only runtime context in parallel without completing formal agents."""
    started = time.perf_counter()
    payload = payload or getattr(nodes, "payload", {}) or {}
    state.setdefault("parallel_read_results", {})
    state.setdefault("parallel_read_warnings", [])
    state.setdefault("parallel_read_branch_timings", {})

    if _is_paused_or_blocked(state):
        state["parallel_read_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return state

    factories = {
        "context_skill": lambda: _context_skill_branch(state, nodes, payload),
    }
    if _should_prepare_rag(state):
        factories["rag_prepare"] = lambda: _rag_prepare_branch(state, payload)

    async def run_one(name: str) -> tuple[str, dict[str, Any] | None, str | None, int]:
        branch_started = time.perf_counter()
        timeout = rag_prepare_timeout if name == "rag_prepare" else context_skill_timeout
        try:
            value = await asyncio.wait_for(factories[name](), timeout=timeout)
            elapsed_ms = int((time.perf_counter() - branch_started) * 1000)
            return name, value, None, elapsed_ms
        except asyncio.TimeoutError:
            elapsed_ms = int((time.perf_counter() - branch_started) * 1000)
            warning = "rag_prepare_timeout" if name == "rag_prepare" else "context_skill_timeout"
            return name, None, warning, elapsed_ms
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - branch_started) * 1000)
            return name, None, f"{name}_failed:{type(exc).__name__}", elapsed_ms

    try:
        completed = await asyncio.wait_for(
            asyncio.gather(*(run_one(name) for name in factories), return_exceptions=True),
            timeout=overall_timeout,
        )
    except asyncio.TimeoutError:
        state.setdefault("parallel_read_warnings", []).append("parallel_read_timeout")
        _ensure_context_fallback(state)
        state["parallel_read_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return state

    for item in completed:
        if isinstance(item, Exception):
            state.setdefault("parallel_read_warnings", []).append(f"parallel_read_failed:{type(item).__name__}")
            continue
        name, value, warning, elapsed_ms = item
        state.setdefault("parallel_read_branch_timings", {})[name] = elapsed_ms
        if warning:
            state.setdefault("parallel_read_warnings", []).append(warning)
            if name == "context_skill":
                _ensure_context_fallback(state)
            elif name == "rag_prepare":
                state.setdefault("parallel_read_results", {})["rag_prepare"] = {
                    "status": "failed",
                    "evidence": [],
                    "evidence_count": 0,
                    "search_attempted": False,
                    "elapsed_ms": elapsed_ms,
                    "summary": "RAG evidence preparation failed.",
                }
            continue
        if name == "context_skill" and value:
            _merge_context_skill_state(state, value)
        elif name == "rag_prepare" and value:
            state.setdefault("parallel_read_results", {})["rag_prepare"] = value

    _ensure_context_fallback(state)
    state["parallel_read_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return state


async def _context_skill_branch(
    state: AgentRuntimeState,
    nodes: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    branch_state = _copy_state(state)
    with SessionLocal() as branch_db:
        branch_nodes = nodes.__class__(branch_db, dict(payload or {}))
        branch_state = await branch_nodes.context_builder(branch_state)
        branch_state = await branch_nodes.skill_matcher(branch_state)
    return branch_state


async def _rag_prepare_branch(state: AgentRuntimeState, payload: dict[str, Any]) -> dict[str, Any]:
    branch_started = time.perf_counter()
    prefetch_rag = (state.get("prefetch_results") or {}).get("rag") or {}
    evidence = list(prefetch_rag.get("evidence") or []) if isinstance(prefetch_rag, dict) else []
    from_prefetch = bool(evidence)
    search_attempted = from_prefetch

    if not evidence:
        user_id = state.get("user_id")
        if user_id is None:
            evidence = []
        else:
            search_attempted = True
            evidence = await asyncio.to_thread(
                rag_service.search_evidence,
                user_id,
                state.get("user_input", "") or state.get("query", ""),
                5,
                0.3,
                _document_ids_for_search(state, payload),
            )
            evidence = list(evidence or [])

    elapsed_ms = int((time.perf_counter() - branch_started) * 1000)
    return {
        "status": "ok",
        "evidence": evidence,
        "evidence_count": len(evidence),
        "sources": _sources_from_evidence(evidence),
        "chunks": _chunks_from_evidence(evidence),
        "elapsed_ms": elapsed_ms,
        "from_prefetch": from_prefetch,
        "search_attempted": search_attempted,
        "summary": "Prepared RAG evidence for downstream rag_agent.",
    }


def _is_paused_or_blocked(state: AgentRuntimeState) -> bool:
    return (
        state.get("status") == "waiting_approval"
        or state.get("route") in {"approval", "blocked"}
        or bool(state.get("approval_payload") and state.get("pending_tool_call_id"))
    )


def _should_prepare_rag(state: AgentRuntimeState) -> bool:
    route = list((state.get("route_plan") or {}).get("route", []) or [])
    if "rag_agent" not in route:
        return False
    intent = str((state.get("route_plan") or {}).get("intent") or state.get("route") or "")
    return intent not in {"tool", "tool.email", "tool.local_file", "tool.browser", "tool.comment", "tool.form_submit", "tool.shell_readonly", "tool.shell_write", "tool.dangerous", "artifact", "memory", "memory_confirm", "research", "feed_research"}


def _copy_state(state: AgentRuntimeState) -> AgentRuntimeState:
    copied: dict[str, Any] = {}
    for key, value in dict(state).items():
        if key == "_stream_queue":
            copied[key] = value
            continue
        try:
            copied[key] = copy.deepcopy(value)
        except Exception:
            copied[key] = value
    return copied  # type: ignore[return-value]


def _merge_context_skill_state(state: AgentRuntimeState, branch_state: dict[str, Any]) -> None:
    for key in (
        "context",
        "context_packet",
        "context_packets",
        "structured_context",
        "context_sections",
        "context_builder_result",
        "matched_skill",
        "matched_skills",
        "candidate_skills",
        "skill_match_result",
        "rag_evidence",
        "conversation_recall_context",
        "memory_context",
        "pipeline_steps",
    ):
        if key in branch_state:
            state[key] = branch_state[key]

    for key in ("agent_outputs", "visible_thoughts", "events"):
        _append_new_items(state, branch_state, key)

    _merge_langgraphstatus_steps(state, branch_state)


def _append_new_items(state: AgentRuntimeState, branch_state: dict[str, Any], key: str) -> None:
    before = list(state.get(key) or [])
    after = list(branch_state.get(key) or [])
    for item in after[len(before):]:
        if item not in before:
            state.setdefault(key, []).append(item)


def _merge_langgraphstatus_steps(state: AgentRuntimeState, branch_state: dict[str, Any]) -> None:
    branch_status = branch_state.get("langgraphstatus")
    if not isinstance(branch_status, dict):
        return
    branch_steps = list(branch_status.get("steps") or [])
    if not branch_steps:
        return
    current = state.setdefault("langgraphstatus", {})
    current_steps = list(current.get("steps") or [])
    existing_keys = {str(item.get("key") or "") for item in current_steps if isinstance(item, dict)}
    for step in branch_steps:
        key = str(step.get("key") or "") if isinstance(step, dict) else ""
        if key and key not in existing_keys:
            current_steps.append(step)
            existing_keys.add(key)
    current["steps"] = current_steps
    state["langgraphstatus"] = current


def _ensure_context_fallback(state: AgentRuntimeState) -> None:
    if not isinstance(state.get("context"), dict):
        state["context"] = {}


def _document_ids_for_search(state: AgentRuntimeState, payload: dict[str, Any]) -> list[int] | None:
    page_context = state.get("page_context") or {}
    raw = payload.get("attachment_ids") or page_context.get("attachment_ids") or None
    if not raw:
        return None
    try:
        return [int(item) for item in raw]
    except (TypeError, ValueError):
        return None


def _sources_from_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        key = (str(item.get("document_id", "")), str(item.get("source_title") or item.get("source_name") or ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "document_id": item.get("document_id"),
            "source_title": item.get("source_title") or item.get("source_name"),
            "source_url": item.get("source_url"),
        })
    return sources


def _chunks_from_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for item in evidence:
        chunks.append({
            "chunk_id": item.get("chunk_id") or item.get("id"),
            "document_id": item.get("document_id"),
            "score": item.get("score"),
        })
    return chunks
