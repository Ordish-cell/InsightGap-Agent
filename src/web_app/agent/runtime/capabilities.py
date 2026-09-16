"""Governed services returning typed observations, never choosing the next action."""

import asyncio
import re
from copy import deepcopy

from sqlalchemy.orm import Session
from langgraph.errors import GraphInterrupt

from src.web_app.agent.runtime.chat_control import check_active, disable_control
from src.web_app.services.rag_service import rag_service
from src.web_app.services.memory_service import memory_service
from src.web_app.services.skill_service import skill_service
from .state import CapabilityResult
from .finalization import emit


async def db_call(nodes, function, *args, **kwargs):
    """Do blocking work off-loop with a session owned by that worker."""
    bind = nodes.db.get_bind()

    def invoke():
        check_active()
        with Session(bind=bind) as db:
            return function(*args, db=db, **kwargs)

    return await asyncio.to_thread(invoke)


def observe(nodes, state, result):
    rows = state.setdefault("observations", [])
    if not any(r["action_id"] == result.action_id for r in rows):
        observation = deepcopy(result.model_dump())
        offset = sum(len(r.get("evidence", [])) for r in rows)
        remap = {}
        for index, evidence in enumerate(observation["evidence"], offset + 1):
            old = evidence.get("evidence_id")
            evidence["evidence_id"] = f"E{index}"
            if old:
                remap[old] = evidence["evidence_id"]

        def mapped(text):
            return re.sub(
                r"\[(E\d+)\]", lambda m: "[" + remap.get(m[1], m[1]) + "]", text
            )

        observation["summary"] = mapped(observation["summary"])
        if observation["data"].get("answer"):
            observation["data"]["answer"] = mapped(observation["data"]["answer"])
            observation["data"]["evidence"] = observation["evidence"]
        rows.append(observation)
    budget = state["runtime_budget"]
    budget["consecutive_failures"] = (
        budget["consecutive_failures"] + 1
        if result.status in {"failed", "unknown"}
        else 0
    )
    state.setdefault("final_warnings", []).extend(result.warnings)
    emit(
        nodes,
        state,
        "capability_result",
        {
            "action_id": result.action_id,
            "capability": result.capability,
            "status": result.status,
            "summary": result.summary,
            "warnings": result.warnings,
        },
    )
    return state


async def execute_capability(nodes, state, runtime_config=None):
    action = state["current_action"]
    kind, args, action_id = (
        action["action"],
        action.get("arguments", {}),
        action["action_id"],
    )
    if any(r["action_id"] == action_id for r in state.get("observations", [])):
        return state
    if kind == "invalid":
        # No I/O or extra failure count; the same Supervisor repairs its native call.
        state.setdefault("observations", []).append(
            CapabilityResult(
                action_id=action_id,
                capability="protocol",
                status="failed",
                error=state.get("native_protocol_error") or "invalid_native_tool_call",
            ).model_dump()
        )
        return state
    disable_control(nodes.db, state["run_id"])
    try:
        if kind == "rag":
            scope = state.get("request", {}).get("attachment_ids") or state.get(
                "page_context", {}
            ).get("attachment_ids")
            ids = args.get("document_ids")
            if scope:
                if ids is not None and not set(ids).issubset(set(scope)):
                    raise ValueError("document_scope_mismatch")
                ids = ids if ids is not None else scope
            if ids is not None:
                from src.web_app.db.repositories.document_repository import (
                    DocumentRepository,
                )

                if any(
                    DocumentRepository(nodes.db).get_by_id_for_user(state["user_id"], i)
                    is None
                    for i in ids
                ):
                    raise ValueError("document_scope_unavailable")
            function = (
                rag_service.ask_document
                if args.get("overview") and ids
                else rag_service.ask
            )
            kw = {"document_ids": ids, "top_k": args["top_k"]}
            if function == rag_service.ask_document:
                kw["overview_mode"] = True
            data = await db_call(nodes, function, state["user_id"], args["query"], **kw)
            state["rag_result"] = data
            status = {"empty": "empty", "failed": "failed", "degraded": "degraded"}.get(
                data.get("retrieval_status"), "ok" if data.get("evidence") else "empty"
            )
            result = CapabilityResult(
                action_id=action_id,
                capability=kind,
                status=status,
                data=data,
                summary=data.get("answer", ""),
                evidence=data.get("evidence", []),
                retryable=status == "failed",
            )
        elif kind == "memory_search":
            items = await db_call(
                nodes,
                memory_service.search_memory,
                state["user_id"],
                args["query"],
                limit=args["top_k"],
            )
            result = CapabilityResult(
                action_id=action_id,
                capability=kind,
                status="ok" if items else "empty",
                data={"memories": items},
            )
        elif kind == "skill" and args["operation"] == "match":
            data = await db_call(
                nodes,
                skill_service.match_skill,
                args["query"] or state["user_input"],
                state["user_id"],
                context=state.get("context", {}),
            )
            state["matched_skill"] = data.get("matched_skill")
            state["candidate_skills"] = data.get("candidate_skills", [])
            data = {**data, "operation": "match"}
            result = CapabilityResult(
                action_id=action_id, capability=kind, status="ok", data=data
            )
        elif kind == "deep_research":
            if not state.get("research_authorized"):
                result = CapabilityResult(
                    action_id=action_id,
                    capability=kind,
                    status="blocked",
                    error="research_confirmation_required",
                    summary="尚未启动深度研究；请解释研究价值并征求用户确认。",
                )
            else:
                from src.web_app.services.research_service import research_service

                if state.get("research_confirmed"):
                    args = {**args, "query": state["research_query"]}
                data = await research_service.research_for_supervisor(
                    nodes.db, state, args, runtime_config=runtime_config
                )
                state["research_result"] = data
                result = CapabilityResult(
                    action_id=action_id,
                    capability=kind,
                    status="ok" if data.get("status") == "completed" else "failed",
                    data=data,
                    summary=data.get("summary", ""),
                    evidence=data.get("evidence", []),
                    error=data.get("error", ""),
                )
        elif kind in {"tool", "artifact", "memory_write", "skill"}:
            from .tools import execute_tool

            mapped = {
                "artifact": "artifact_mcp.create_text_artifact",
                "memory_write": "memory_mcp.add",
                "skill": "skill_mcp.create_draft",
            }
            name = args["name"] if kind == "tool" else mapped[kind]
            inputs = (
                args["input"]
                if kind == "tool"
                else args["draft"]
                if kind == "skill"
                else args
            )
            result = await execute_tool(nodes, state, name, inputs)
        else:
            raise ValueError("unknown_capability")
    except GraphInterrupt:
        raise
    except Exception as exc:
        result = CapabilityResult(
            action_id=action_id,
            capability=kind,
            status="failed",
            error=type(exc).__name__ + ": " + str(exc)[:200],
        )
    return observe(nodes, state, result)
