"""Tool intents cross a deterministic policy boundary before any side effect."""

import asyncio
import json
import re

from langgraph.types import interrupt

from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.mcp_repository import ToolCallRepository
from src.web_app.mcp.registry import normalize_tool_name
from src.web_app.mcp.tool_router import validate_tool_input
from src.web_app.mcp.tool_executor import tool_executor, hash_tool_args
from src.web_app.mcp.audit import _redact_sensitive
from src.web_app.services.mcp_service import mcp_service
from .state import CapabilityResult
from .capabilities import db_call
from .finalization import emit
from .policy import public_result


def save_allowed(state, name):
    if name in {"memory_mcp.add", "memory_mcp.extract"} and state.get("basic_memory"):
        from src.web_app.memory.basic_facts import authorized_fact, EXPLICIT_CONFIRMATIONS
        plan = state["basic_memory"]
        if plan.get("blocked") or (not plan.get("facts") and state["user_input"].strip().rstrip("。！.! ") in EXPLICIT_CONFIRMATIONS):
            return False
        if plan.get("facts"):
            grant = authorized_fact.get()
            return bool(name == "memory_mcp.add" and grant and grant["run_id"] == state["run_id"] and grant["user_id"] == state["user_id"])
    if name == "memory_mcp.extract":
        name = "memory_mcp.add"
    rules = {
        "artifact_mcp.create_text_artifact": (
            "save_artifact",
            "artifact",
            r"保存|生成.*(?:报告|文件|文档)|导出|save|export|create.*(?:file|report|document)",
        ),
        "memory_mcp.add": (
            "write_memory",
            "memory",
            r"记住|记一下|记下来|以后.*(?:偏好|喜欢|使用)|remember|memorize",
        ),
        "skill_mcp.create_draft": (
            "create_skill_draft",
            "skill",
            r"(?:生成|创建|保存).*(?:技能|skill)|(?:create|save).*skill",
        ),
    }
    if name not in rules:
        return True
    flag, route, pattern = rules[name]
    request = state.get("request", {})
    if state.get("save_policy", {}).get(flag) or request.get("route") == route:
        return True
    text = state["user_input"]
    return not re.search(r"不要|不用|不必|don't|do not", text, re.I) and bool(
        re.search(pattern, text, re.I)
    )


async def execute_tool(nodes, state, name, inputs):
    action = state["current_action"]
    identity = {"action_id": action["action_id"], "capability": action["action"]}
    name = normalize_tool_name(name)
    if name in {"context.graph", "context.history"}:
        query = str(inputs.get("query") or state["user_input"])
        if name == "context.graph":
            from src.web_app.services.graph_context_service import graph_context_service

            data = await asyncio.to_thread(
                graph_context_service.get_context,
                user_id=state["user_id"],
                query=query,
                route="rag",
            )
        else:
            from src.web_app.services.conversation_summary_service import (
                conversation_summary_service,
            )

            rows = await db_call(
                nodes,
                conversation_summary_service.search_relevant_segments,
                conversation_id=state["conversation_id"],
                query=query,
                user_id=state["user_id"],
            )
            data = [getattr(r, "summary_text", str(r)) for r in rows]
        return CapabilityResult(
            **identity, status="ok" if data else "empty", data={"context": data}
        )
    spec = mcp_service.get_tool(nodes.db, name)
    if spec:
        state["risk_level"] = str(spec.get("permission_level") or spec.get("safety_level", "L0")).split("_")[0]
    if not spec or not spec.get("enabled", True):
        return CapabilityResult(
            **identity, status="blocked", error="tool_not_found_or_disabled"
        )
    # A refusal ends write authority for this turn, including alternative tools.
    risk = spec.get("permission_level") or spec.get("safety_level", "")
    if str(risk).startswith("L3") and not getattr(nodes, "checkpoint_enabled", False):
        return CapabilityResult(
            **identity, status="blocked", error="approval_checkpoint_unavailable"
        )
    if state.get("writes_denied") and not str(risk).startswith("L0"):
        return CapabilityResult(
            **identity, status="blocked", error="write_authority_revoked"
        )
    if not save_allowed(state, name):
        return CapabilityResult(
            **identity, status="blocked", error="explicit_save_intent_required"
        )
    cleaned, missing = validate_tool_input(name, inputs)
    if missing:
        return CapabilityResult(
            **identity,
            status="empty",
            error="missing_fields",
            data={"missing_fields": missing},
            summary="工具尚未执行，需要补充参数。",
        )
    from jsonschema import Draft202012Validator

    errors = list(
        Draft202012Validator(spec.get("input_schema") or {}).iter_errors(cleaned)
    )
    if errors:
        return CapabilityResult(
            **identity,
            status="failed",
            error="invalid_tool_arguments",
            data={"fields": [list(e.path) for e in errors]},
        )
    key = (
        f"run:{state['run_id']}:action:{action['action_id']}:{hash_tool_args(cleaned)}"
    )
    if ToolCallRepository(nodes.db).get_by_idempotency_key(key) is None:
        emit(
            nodes,
            state,
            "tool_call_started",
            {
                "tool_call_id": action["action_id"],
                "tool_name": name,
                "args_preview": public_result(_redact_sensitive(cleaned)),
                "status": "running",
            },
        )
    record = await db_call(
        nodes,
        mcp_service.call_tool,
        user_id=state["user_id"],
        tool_name=name,
        input_data=cleaned,
        agent_run_id=state["run_id"],
        idempotency_key=key,
        approval_mode="agent_runtime",
        dry_run=bool(state.get("request", {}).get("dry_run", False)),
    )
    if record["status"] == "waiting_approval":
        approval_id = record.get("approval_id") or record.get("output", {}).get(
            "_metadata", {}
        ).get("approval_id")
        approval = ApprovalRepository(nodes.db).get_by_user(
            state["user_id"], int(approval_id)
        )
        if not approval:
            raise ValueError("approval_missing")
        from src.web_app.services.agent_service import _sanitize_tool_args_for_frontend

        payload = {
            "type": "approval_required",
            "approval_pause_mode": "interrupt",
            "run_id": state["run_id"],
            "approval_id": approval.id,
            "tool_call_id": record["id"],
            "tool_name": name,
            "risk_level": approval.payload.get("risk_level", risk),
            "tool_args": _sanitize_tool_args_for_frontend(name, cleaned),
            "preview": approval.payload.get("preview", {}),
            "actions": ["approve", "reject"],
        }
        resumed = interrupt(payload)
        # LangGraph restarts this node. Preparation above is keyed by persisted action_id.
        approval = ApprovalRepository(nodes.db).get_by_user(
            state["user_id"], int(approval_id)
        )
        nodes.db.refresh(approval)
        if int(resumed.get("tool_call_id") or 0) != record["id"] or (
            resumed.get("approval_id")
            and str(resumed["approval_id"]) != str(approval.id)
        ):
            raise ValueError("approval_resume_mismatch")
        if approval.status == "rejected" and resumed.get("action") == "rejected":
            ToolCallRepository(nodes.db).update_status(
                record["id"], "rejected", error_message="User rejected the approval"
            )
            state["writes_denied"] = True
            record = {
                **record,
                "status": "rejected",
                "error": "User rejected the approval",
            }
        elif approval.status == "approved" and resumed.get("action") == "approved":
            output = await db_call(
                nodes,
                tool_executor.execute_approved_tool_once,
                user_id=state["user_id"],
                tool_call_id=record["id"],
                tool_name=name,
                input_data=cleaned,
                agent_run_id=state["run_id"],
            )
            record = {
                **record,
                "output": output,
                "status": "completed"
                if output.get("success")
                else "unknown"
                if output.get("error_code") == "TOOL_OUTCOME_UNKNOWN"
                else "failed",
                "error": output.get("message", "") if not output.get("success") else "",
            }
        else:
            raise ValueError("approval_not_resumable")
    state.update(
        approval_required=False,
        approval_payload=None,
        pending_approval_id=None,
        pending_tool_call_id=None,
        pending_tool_name=None,
        pending_tool_args=None,
        resume_token=None,
        status="running",
    )
    state["tool_call"] = record
    state["tool_result"] = record
    calls = state.setdefault("tool_calls", [])
    if not any(r.get("id") == record.get("id") for r in calls):
        calls.append(record)
    output = record.get("output") or {}
    status = {
        "completed": "ok",
        "rejected": "rejected",
        "blocked": "blocked",
        "running": "unknown",
        "pending": "unknown",
        "unknown": "unknown",
    }.get(record["status"], "failed")
    if output.get("success") is False and status not in {
        "unknown",
        "blocked",
        "rejected",
    }:
        status = "failed"
    if name == "memory_mcp.add" and (output.get("memory") or {}).get("ok") is False:
        status = "failed"
    if status == "unknown":
        state["writes_denied"] = True
    if status == "ok" and name == "web.search" and not output.get("results"):
        status = "failed" if output.get("error") else "empty"
    if status == "ok" and (
        name in {"search_mcp.search", "github_mcp.repo_summary"}
        or output.get("provider") == "mock"
        or output.get("dry_run")
    ):
        status = "degraded"
    if status == "ok":
        if output.get("artifact_id"):
            state.setdefault("artifacts", []).append(
                {"id": output["artifact_id"], **output}
            )
        if output.get("memory_id"):
            state.setdefault("memory_updates", []).append(output["memory"])
            state.setdefault("memory_save_results", []).append(output["memory"])
        if output.get("skill_id"):
            state.setdefault("skill_drafts", []).append(output["skill"])
        if name == "memory_mcp.extract":
            receipts = output.get("save_results", [])
            state.setdefault("memory_save_results", []).extend(receipts)
            state.setdefault("memory_updates", []).extend(
                r for r in receipts if r.get("ok")
            )
            if not receipts:
                status = "empty"
            elif not any(r.get("ok") for r in receipts):
                status = "failed"
    emit(
        nodes,
        state,
        "tool_call_completed"
        if status in {"ok", "degraded", "empty"}
        else "tool_call_failed",
        {
            "tool_call_id": action["action_id"],
            "tool_call_record_id": record["id"],
            "tool_name": name,
            "output_preview": json.dumps(public_result(_redact_sensitive(output)), ensure_ascii=False, default=str)[:500],
            "status": record["status"],
        },
    )
    evidence = (
        [
            {
                "source_title": r.get("title", ""),
                "url": r.get("url", ""),
                "quote": r.get("snippet", ""),
                "source_type": "web",
            }
            for r in output.get("results", [])
            if isinstance(r, dict) and r.get("url")
        ]
        if name == "web.search"
        else []
    )
    return CapabilityResult(
        **identity,
        status=status,
        data=record,
        evidence=evidence,
        summary=str(
            output.get("summary") or output.get("message") or f"{name}: {status}"
        ),
        error=record.get("error", ""),
    )
