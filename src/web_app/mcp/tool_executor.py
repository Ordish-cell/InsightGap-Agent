import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.mcp_repository import MCPToolRepository, ToolCallRepository
from src.web_app.mcp.audit import tool_call_to_read
from src.web_app.mcp.local_provider import local_provider
from src.web_app.mcp.registry import registry
from src.web_app.mcp.schemas import ToolCallRead
from src.web_app.services.permission_service import PermissionGuard

logger = logging.getLogger(__name__)


class MCPToolExecutor:
    def call_tool(
        self,
        db: Session,
        user_id: int,
        tool_name: str,
        input_data: dict[str, Any],
        agent_run_id: int | None = None,
        dry_run: bool = False,
        idempotency_key: str | None = None,
        approval_mode: str = "standalone",
    ) -> ToolCallRead:
        registry.ensure_builtin_tools(db)
        tool = MCPToolRepository(db).get_by_name(tool_name)
        safety_level = tool.permission_level if tool else L4_HIGH_RISK
        idempotency_key = idempotency_key or (
            build_tool_idempotency_key(agent_run_id, user_id, tool_name, input_data)
            if agent_run_id is not None
            else None
        )

        call_repo = ToolCallRepository(db)
        if idempotency_key:
            existing_call = call_repo.get_by_idempotency_key(idempotency_key)
            if existing_call:
                approval_id = _approval_id_for_call(db, existing_call)
                return tool_call_to_read(existing_call, approval_id=approval_id)

        call = call_repo.create(
            user_id=user_id,
            run_id=agent_run_id,
            tool_name=tool_name,
            mcp_tool_id=tool.id if tool else None,
            input=input_data,
            output={},
            permission_level=safety_level,
            status="pending",
            error_message="",
            idempotency_key=idempotency_key,
        )

        if not tool or not tool.enabled:
            return self._finish(db, call, "failed", {}, "Tool not found or disabled")

        decision = PermissionGuard().check_tool_call(tool_name, tool.permission_level, tool.approval_required)
        if tool.permission_level == L4_HIGH_RISK or (not decision["allowed"] and not decision["requires_approval"]):
            logger.info("[APPROVAL_FLOW] tool_executor blocked tool=%s risk=%s user=%s", tool_name, safety_level, user_id)
            return self._finish(db, call, "blocked", {}, decision["reason"])

        if tool.permission_level == L3_EXTERNAL_WRITE or decision["requires_approval"]:
            logger.info("[APPROVAL_FLOW] tool_executor prepared approval tool=%s user=%s run_id=%s", tool_name, user_id, agent_run_id)
            preview = _build_action_preview(tool_name, input_data)
            safety_notes = _build_safety_notes(tool_name, tool.permission_level)
            approval_payload = {
                "tool_call_id": call.id,
                "tool_name": tool_name,
                "tool_args": input_data,
                "risk_level": tool.permission_level,
                "preview": preview,
                "safety_notes": safety_notes,
                "requires_approval": True,
                "approval_mode": approval_mode,
                "idempotency_key": idempotency_key,
            }
            approval = self._get_or_create_approval(
                db,
                user_id=user_id,
                agent_run_id=agent_run_id,
                idempotency_key=idempotency_key,
                tool_name=tool_name,
                input_data=input_data,
                risk_level=tool.permission_level,
                payload=approval_payload,
            )
            return self._finish(
                db,
                call,
                "waiting_approval",
                {"_metadata": {"approval_id": approval.id, "waiting_approval": True}},
                "approval_required",
                approval_id=approval.id,
            )

        if dry_run:
            return self._finish(db, call, "completed", {"dry_run": True, "would_call": tool_name}, "")

        call_repo.update(call, status="running")
        try:
            output = local_provider.call(db, user_id, tool_name, input_data, agent_run_id)
            return self._finish(db, call, "completed", output, "")
        except Exception as exc:
            return self._finish(db, call, "failed", {}, str(exc))

    def execute_approved_tool(
        self,
        db: Session,
        user_id: int,
        tool_call_id: int,
        tool_name: str,
        input_data: dict[str, Any],
        agent_run_id: int | None = None,
    ) -> dict[str, Any]:
        return self.execute_approved_tool_once(db, user_id, tool_call_id, tool_name, input_data, agent_run_id=agent_run_id)

    def execute_approved_tool_once(
        self,
        db: Session,
        user_id: int,
        tool_call_id: int,
        tool_name: str,
        input_data: dict[str, Any],
        agent_run_id: int | None = None,
    ) -> dict[str, Any]:
        repo = ToolCallRepository(db)
        call = repo.get_by_user(user_id, tool_call_id)
        if not call:
            raise ValueError(f"ToolCall not found: {tool_call_id}")
        if call.status == "completed":
            return _normalize_tool_output(tool_name, call.output or {})
        if call.status in {"blocked", "rejected"}:
            return {
                "success": False,
                "tool_name": tool_name,
                "error_code": f"TOOL_CALL_{call.status.upper()}",
                "message": call.error_message or f"Tool call is {call.status}.",
            }

        repo.update_status(tool_call_id, "running")
        try:
            output = local_provider.call(db, user_id, tool_name, input_data, agent_run_id)
            repo.update_status(tool_call_id, "completed", output=output)
            return _normalize_tool_output(tool_name, output)
        except Exception as exc:
            repo.update_status(tool_call_id, "failed", error_message=str(exc))
            return {
                "success": False,
                "tool_name": tool_name,
                "error_code": type(exc).__name__,
                "message": str(exc),
            }

    def _finish(self, db: Session, call, status: str, output: dict[str, Any], error: str, approval_id: int | None = None) -> ToolCallRead:
        completed_at = datetime.now(UTC).replace(tzinfo=None)
        output = dict(output or {})
        output.setdefault("_metadata", {})
        output["_metadata"]["completed_at"] = completed_at.isoformat()
        if approval_id:
            output["_metadata"]["approval_id"] = approval_id
        updated = ToolCallRepository(db).update(call, status=status, output=output, error_message=error)
        return tool_call_to_read(updated, approval_id=approval_id, completed_at=completed_at)

    def _get_or_create_approval(
        self,
        db: Session,
        *,
        user_id: int,
        agent_run_id: int | None,
        idempotency_key: str | None,
        tool_name: str,
        input_data: dict[str, Any],
        risk_level: str,
        payload: dict[str, Any],
    ):
        repo = ApprovalRepository(db)
        if idempotency_key:
            existing = repo.get_by_idempotency_key(idempotency_key)
            if existing:
                return existing
        return repo.create(
            user_id=user_id,
            run_id=agent_run_id,
            approval_type="mcp_tool_call",
            title=f"Approval required: {_tool_display_name(tool_name)}",
            description=_build_approval_description(tool_name, input_data, risk_level),
            payload=payload,
            idempotency_key=idempotency_key,
        )


def canonical_tool_args(input_data: dict[str, Any]) -> str:
    return json.dumps(input_data or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def hash_tool_args(input_data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_tool_args(input_data).encode("utf-8")).hexdigest()


def build_tool_idempotency_key(agent_run_id: int | None, user_id: int, tool_name: str, input_data: dict[str, Any]) -> str:
    scope = f"agent_run:{agent_run_id}" if agent_run_id is not None else f"user:{user_id}"
    return f"{scope}:{tool_name}:{hash_tool_args(input_data)}"


def _normalize_tool_output(tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {
            "success": False,
            "tool_name": tool_name,
            "error_code": "EMPTY_TOOL_RESULT",
            "message": "Tool returned no structured result.",
        }
    success = output.get("success", None)
    if success is None:
        success = output.get("sent", output.get("written", None))
    return {
        "success": success if isinstance(success, bool) else True,
        "tool_name": tool_name,
        **{k: v for k, v in output.items() if k not in ("_metadata",)},
    }


def _approval_id_for_call(db: Session, call) -> int | None:
    metadata = (call.output or {}).get("_metadata", {}) if isinstance(call.output, dict) else {}
    approval_id = metadata.get("approval_id")
    if approval_id:
        return approval_id
    if call.idempotency_key:
        approval = ApprovalRepository(db).get_by_idempotency_key(call.idempotency_key)
        return approval.id if approval else None
    return None


def _tool_display_name(tool_name: str) -> str:
    names = {
        "email.send": "Send email",
        "local_file.write": "Write file",
        "local_file.append": "Append file",
        "local_file.delete": "Delete file",
        "local_file.read": "Read file",
        "local_file.list": "List files",
        "web.search": "Web search",
        "system.time": "Local time",
        "system.calc": "Calculator",
        "system.unit_convert": "Unit conversion",
        "system.uuid": "Generate UUID",
        "system.hash": "Hash calculation",
        "artifact_mcp.create_text_artifact": "Create artifact",
        "browser_mcp.plan_actions": "Plan browser actions",
    }
    return names.get(tool_name, tool_name)


def _build_action_preview(tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "email.send":
        return {
            "title": "Send email",
            "to": input_data.get("to", ""),
            "subject": input_data.get("subject", ""),
            "body": str(input_data.get("body", ""))[:500],
        }
    if tool_name in ("local_file.write", "local_file.append"):
        return {
            "title": "Write file" if tool_name == "local_file.write" else "Append file",
            "path": input_data.get("path", ""),
            "content_preview": str(input_data.get("content", ""))[:300],
            "chars": len(str(input_data.get("content", ""))),
        }
    if tool_name == "local_file.delete":
        return {"title": "Delete file", "path": input_data.get("path", ""), "warning": "High-risk irreversible action."}
    if tool_name in ("local_file.read", "local_file.list"):
        return {"title": "Read file" if tool_name == "local_file.read" else "List files", "path": input_data.get("path", ".")}
    if tool_name == "web.search":
        return {"title": "Web search", "query": input_data.get("query", ""), "limit": input_data.get("limit", 5)}
    if tool_name.startswith("system."):
        return {"title": _tool_display_name(tool_name), "args": {k: str(v)[:200] for k, v in input_data.items()}}
    return {"title": _tool_display_name(tool_name), "args": {k: str(v)[:200] for k, v in input_data.items()}}


def _build_safety_notes(tool_name: str, risk_level: str) -> list[str]:
    notes: list[str] = []
    if risk_level == L3_EXTERNAL_WRITE:
        notes.append("External write action. It requires approval before execution.")
    if "email" in tool_name:
        notes.append("This may send real information to an external recipient.")
    if "local_file.write" in tool_name:
        notes.append("The file will be written under the configured safe workspace.")
    if "local_file.append" in tool_name:
        notes.append("The content will be appended to the target file.")
    if risk_level == L4_HIGH_RISK:
        notes.append("High-risk action. It is blocked by default.")
    return notes


def _build_approval_description(tool_name: str, input_data: dict[str, Any], risk_level: str) -> str:
    if tool_name == "email.send":
        return f"To: {input_data.get('to', '')}\nSubject: {input_data.get('subject', '')}\nRisk: {risk_level}"
    if tool_name in ("local_file.write", "local_file.append"):
        return f"Path: {input_data.get('path', '')}\nRisk: {risk_level}"
    return f"Tool: {tool_name}\nRisk: {risk_level}"


tool_executor = MCPToolExecutor()
