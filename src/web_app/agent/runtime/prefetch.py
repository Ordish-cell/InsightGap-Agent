from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.schemas import AgentResult, append_agent_result
from src.web_app.agent.runtime.state import AgentRuntimeState
from src.web_app.db.session import SessionLocal
from src.web_app.services.memory_service import memory_service
from src.web_app.services.rag_service import rag_service
from src.web_app.services.skill_service import skill_service

PREFETCH_TIMEOUT_SECONDS = 3.0


async def parallel_prefetch(
    state: AgentRuntimeState,
    db: Session,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = PREFETCH_TIMEOUT_SECONDS,
) -> AgentRuntimeState:
    """Run low-risk read-only prefetch tasks without blocking the main route."""
    started = time.perf_counter()
    user_id = state.get("user_id")
    query = state.get("user_input", "") or state.get("query", "")
    route = state.get("route", "chat")
    answer_mode = str((state.get("route_plan") or {}).get("answer_mode") or state.get("answer_mode") or "")
    if answer_mode == "conversation_recall":
        memory_factory = lambda: _memory_prefetch(user_id, query, db, answer_mode=answer_mode)
    else:
        memory_factory = lambda: _memory_prefetch(user_id, query, db)

    tasks: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {
        "rag": lambda: _rag_prefetch(user_id, query),
        "memory": memory_factory,
        "skill": lambda: _skill_prefetch(user_id, query, db, state, payload),
        "graph": lambda: _graph_prefetch(user_id, query, route),
    }

    results: dict[str, Any] = {}
    warnings: list[str] = []
    agent_results: list[dict[str, Any]] = []

    async def run_one(name: str, factory: Callable[[], Awaitable[dict[str, Any]]]) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            value = await asyncio.wait_for(factory(), timeout=timeout_seconds)
            return name, value, None
        except asyncio.TimeoutError:
            return name, None, f"prefetch_timeout:{name}"
        except Exception as exc:
            return name, None, f"{name}_prefetch_failed:{type(exc).__name__}"

    completed = await asyncio.gather(
        *(run_one(name, factory) for name, factory in tasks.items()),
        return_exceptions=True,
    )

    for item in completed:
        if isinstance(item, Exception):
            name, value, warning = "unknown", None, f"prefetch_failed:{type(item).__name__}"
        else:
            name, value, warning = item
        if warning:
            warnings.append(warning)
            agent_result = AgentResult(
                task_id=f"prefetch:{name}",
                agent=f"{name}_prefetch",
                status="failed",
                confidence=0.0,
                summary=f"{name} prefetch failed",
                warnings=[warning],
                metadata={"source": "prefetch"},
            ).model_dump()
        else:
            results[name] = value or {}
            agent_result = _agent_result_for_prefetch(name, value or {}).model_dump()
        agent_results.append(agent_result)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    state["prefetch_results"] = results
    state["prefetch_warnings"] = warnings
    state["prefetch_elapsed_ms"] = elapsed_ms
    state["prefetch_agent_results"] = agent_results
    for result in agent_results:
        append_agent_result(state, result)
    return state


async def _rag_prefetch(user_id: int | None, query: str) -> dict[str, Any]:
    if user_id is None:
        return {"evidence": [], "count": 0, "skipped": True}
    evidence = await asyncio.to_thread(
        rag_service.search_evidence,
        user_id,
        query,
        limit=5,
        score_threshold=0.3,
    )
    return {"evidence": evidence or [], "count": len(evidence or [])}


async def _memory_prefetch(user_id: int | None, query: str, db: Session, *, answer_mode: str = "") -> dict[str, Any]:
    if user_id is None:
        return {"items": [], "count": 0, "skipped": True}
    if answer_mode == "conversation_recall":
        return {
            "items": [],
            "count": 0,
            "skipped": True,
            "reason": "conversation_recall_uses_conversation_history_only",
        }
    items = await asyncio.to_thread(_search_memory_in_thread, user_id, query)
    backend = getattr(memory_service, "_last_search_backend", "unknown")
    qdrant_hits = getattr(memory_service, "_last_qdrant_hits", 0)
    return {"items": items or [], "count": len(items or []), "backend": backend, "qdrant_hits": qdrant_hits}


async def _skill_prefetch(user_id: int | None, query: str, db: Session, state: AgentRuntimeState, payload: dict[str, Any]) -> dict[str, Any]:
    if user_id is None or payload.get("use_existing_skills", True) is False or payload.get("auto_skill", True) is False:
        return {"matched_skill": None, "candidate_skills": [], "skipped": True}
    result = await asyncio.to_thread(_match_skill_in_thread, user_id, query, state.get("context", {}))
    return {
        "matched_skill": result.get("matched_skill"),
        "candidate_skills": result.get("candidate_skills", []),
        "raw": result,
    }


async def _graph_prefetch(user_id: int | None, query: str, route: str) -> dict[str, Any]:
    if user_id is None:
        return {"context": "", "debug": {}, "skipped": True}
    try:
        from src.web_app.services.graph_context_service import graph_context_service
    except Exception:
        return {"context": "", "debug": {"warning": "graph_prefetch_unavailable"}, "skipped": True}
    context = await asyncio.to_thread(
        graph_context_service.get_context,
        user_id=user_id,
        query=query,
        route=route,
    )
    debug = getattr(graph_context_service, "last_debug", {}) or {}
    return {"context": context or "", "debug": debug}


def _search_memory_in_thread(user_id: int, query: str) -> list[dict[str, Any]]:
    with SessionLocal() as thread_db:
        return memory_service.search_memory(
            user_id,
            query,
            min_importance=0.2,
            db=thread_db,
            limit=5,
        )


def _match_skill_in_thread(user_id: int, query: str, context: dict[str, Any]) -> dict[str, Any]:
    with SessionLocal() as thread_db:
        return skill_service.match_skill(query, user_id, thread_db, context)


def _agent_result_for_prefetch(name: str, value: dict[str, Any]) -> AgentResult:
    count = int(value.get("count") or len(value.get("candidate_skills", []) or []) or (1 if value.get("context") else 0))
    skipped = bool(value.get("skipped"))
    return AgentResult(
        task_id=f"prefetch:{name}",
        agent=f"{name}_prefetch",
        status="skipped" if skipped else "ok",
        confidence=0.4 if skipped else 0.75,
        summary=f"{name} prefetch {'skipped' if skipped else 'completed'}",
        findings=[f"count={count}"],
        evidence=list(value.get("evidence") or []),
        warnings=["prefetch_skipped"] if skipped else [],
        metadata={"source": "prefetch"},
    )
