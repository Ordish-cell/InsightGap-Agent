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


class MCPToolExecutor:
    def call_tool(self, db: Session, user_id: int, tool_name: str, input_data: dict[str, Any], agent_run_id: int | None = None, dry_run: bool = False) -> ToolCallRead:
        registry.ensure_builtin_tools(db)
        tool = MCPToolRepository(db).get_by_name(tool_name)
        safety_level = tool.permission_level if tool else L4_HIGH_RISK
        call = ToolCallRepository(db).create(user_id=user_id, run_id=agent_run_id, tool_name=tool_name, mcp_tool_id=tool.id if tool else None, input=input_data, output={}, permission_level=safety_level, status="pending", error_message="")

        if not tool or not tool.enabled:
            return self._finish(db, call, "failed", {}, "Tool not found or disabled")

        decision = PermissionGuard().check_tool_call(tool_name, tool.permission_level, tool.approval_required)
        if tool.permission_level == L4_HIGH_RISK or (not decision["allowed"] and not decision["requires_approval"]):
            return self._finish(db, call, "blocked", {}, decision["reason"])
        if tool.permission_level == L3_EXTERNAL_WRITE or decision["requires_approval"]:
            # Build a rich preview for the approval card
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
            }
            approval = ApprovalRepository(db).create(
                user_id=user_id,
                run_id=agent_run_id,
                approval_type="mcp_tool_call",
                title=f"需要你确认：{_tool_display_name(tool_name)}",
                description=_build_approval_description(tool_name, input_data, tool.permission_level),
                payload=approval_payload,
            )
            return self._finish(db, call, "waiting_approval", {"_metadata": {"approval_id": approval.id, "waiting_approval": True}}, "approval_required", approval_id=approval.id)
        if dry_run:
            return self._finish(db, call, "completed", {"dry_run": True, "would_call": tool_name}, "")

        ToolCallRepository(db).update(call, status="running")
        try:
            output = local_provider.call(db, user_id, tool_name, input_data, agent_run_id)
            return self._finish(db, call, "completed", output, "")
        except Exception as exc:
            return self._finish(db, call, "failed", {}, str(exc))

    def execute_approved_tool(self, db: Session, user_id: int, tool_call_id: int, tool_name: str, input_data: dict[str, Any], agent_run_id: int | None = None) -> dict[str, Any]:
        """Execute a tool that has been approved. Bypasses the permission guard.

        Always returns a flat dict with:
          success: True | False
          tool_name: str
          provider: str | None  (mock / smtp / local_file / None)
          message: str
        Plus tool-specific fields (to, subject, body_preview, path, etc.)
        """
        ToolCallRepository(db).update_status(tool_call_id, "running")
        try:
            output = local_provider.call(db, user_id, tool_name, input_data, agent_run_id)
            ToolCallRepository(db).update_status(tool_call_id, "completed", output=output)

            # Normalize to flat dict with success at top level
            if not isinstance(output, dict):
                return {
                    "success": False,
                    "tool_name": tool_name,
                    "error_code": "EMPTY_TOOL_RESULT",
                    "message": "工具没有返回结果，无法确认执行成功。",
                }

            # Flatten: if success is nested inside output, lift it
            success = output.get("success", None)
            if success is None:
                # Check if the outer wrapper has success
                success = output.get("success", output.get("sent", output.get("written", None)))

            return {
                "success": success if isinstance(success, bool) else True,
                "tool_name": tool_name,
                **{k: v for k, v in output.items() if k not in ("_metadata",)},
            }
        except Exception as exc:
            ToolCallRepository(db).update_status(tool_call_id, "failed", error_message=str(exc))
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


def _tool_display_name(tool_name: str) -> str:
    names = {
        "email.send": "发送邮件",
        "local_file.write": "写入文件",
        "local_file.append": "追加文件",
        "local_file.delete": "删除文件",
        "local_file.read": "读取文件",
        "local_file.list": "列出文件",
        "artifact_mcp.create_text_artifact": "创建文档",
        "browser_mcp.plan_actions": "浏览器操作",
    }
    return names.get(tool_name, tool_name)


def _build_action_preview(tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "email.send":
        return {
            "title": "发送邮件",
            "to": input_data.get("to", ""),
            "subject": input_data.get("subject", ""),
            "body": str(input_data.get("body", ""))[:500],
        }
    if tool_name in ("local_file.write", "local_file.append"):
        return {
            "title": "写入文件" if tool_name == "local_file.write" else "追加文件",
            "path": input_data.get("path", ""),
            "content_preview": str(input_data.get("content", ""))[:300],
            "chars": len(str(input_data.get("content", ""))),
        }
    if tool_name == "local_file.delete":
        return {
            "title": "删除文件",
            "path": input_data.get("path", ""),
            "warning": "高危操作：删除文件不可恢复",
        }
    if tool_name in ("local_file.read", "local_file.list"):
        return {
            "title": "读取文件" if tool_name == "local_file.read" else "列出文件",
            "path": input_data.get("path", "."),
        }
    return {"title": _tool_display_name(tool_name), "args": {k: str(v)[:200] for k, v in input_data.items()}}


def _build_safety_notes(tool_name: str, risk_level: str) -> list[str]:
    notes: list[str] = []
    if risk_level == "L3_EXTERNAL_WRITE":
        notes.append("外部写入操作，需要审批后才能执行")
    if "email" in tool_name:
        notes.append("这会向外部联系人发送真实信息")
    if "local_file.write" in tool_name:
        notes.append("文件将被写入到安全工作目录下")
        notes.append("不会覆盖系统文件或项目配置")
    if "local_file.append" in tool_name:
        notes.append("内容将追加到文件末尾")
    if risk_level == "L4_HIGH_RISK":
        notes.append("危险操作，默认阻止")
    return notes


def _build_approval_description(tool_name: str, input_data: dict[str, Any], risk_level: str) -> str:
    if tool_name == "email.send":
        return f"收件人: {input_data.get('to', '未指定')}\n主题: {input_data.get('subject', '')}\n风险等级: {risk_level}"
    if tool_name in ("local_file.write", "local_file.append"):
        return f"路径: {input_data.get('path', '')}\n风险等级: {risk_level}"
    return f"工具: {tool_name}\n风险等级: {risk_level}"


tool_executor = MCPToolExecutor()
