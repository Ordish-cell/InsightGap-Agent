from datetime import UTC, datetime
from typing import Any

from src.web_app.mcp.schemas import ToolCallRead

# ── Sensitive keys that must never appear in logs ──
_SENSITIVE_KEYS = {
    "api_key", "api_secret", "password", "passwd", "secret",
    "token", "auth_token", "access_token", "jwt", "private_key",
    "smtp_password", "credential", "authorization",
}


def _redact_sensitive(data: dict[str, Any]) -> dict[str, Any]:
    """Deep-redact sensitive values from a dictionary."""
    if not isinstance(data, dict):
        return data

    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        lower = key.lower().replace("-", "_").replace(" ", "_")
        if any(sensitive in lower for sensitive in _SENSITIVE_KEYS):
            cleaned[key] = "***REDACTED***"
        elif isinstance(value, dict):
            cleaned[key] = _redact_sensitive(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _redact_sensitive(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _safe_preview(value: Any, max_chars: int = 500) -> str:
    """Truncate a value to a safe preview length."""
    if not value:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [{len(text)} total chars]"


def build_audit_record(
    user_id: int,
    run_id: int | None,
    tool_name: str,
    risk_level: str,
    status: str,
    args: dict[str, Any],
    result: dict[str, Any] | None = None,
    approval_id: int | None = None,
    error_type: str = "",
    error_message: str = "",
    duration_ms: int = 0,
) -> dict[str, Any]:
    """Build a complete audit record for a tool call."""
    safe_args = _redact_sensitive(dict(args))
    safe_result = _redact_sensitive(dict(result or {}))
    return {
        "user_id": user_id,
        "run_id": run_id,
        "tool_name": tool_name,
        "risk_level": risk_level,
        "status": status,
        "args_preview": _safe_preview(safe_args, 300),
        "result_preview": _safe_preview(safe_result, 300),
        "approval_id": approval_id,
        "duration_ms": duration_ms,
        "error_type": error_type,
        "error_message": _safe_preview(error_message, 200),
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def tool_call_to_read(item, approval_id: int | None = None, completed_at: datetime | None = None) -> ToolCallRead:
    output = item.output or {}
    metadata = output.get("_metadata", {}) if isinstance(output, dict) else {}
    # Redact sensitive info from input/output
    safe_input = _redact_sensitive(dict(item.input or {}))
    safe_output = _redact_sensitive({key: value for key, value in output.items() if key != "_metadata"}) if isinstance(output, dict) else {}
    return ToolCallRead(
        id=item.id,
        user_id=item.user_id,
        agent_run_id=item.run_id,
        tool_id=item.mcp_tool_id,
        tool_name=item.tool_name,
        safety_level=item.permission_level,
        status=item.status,
        input=safe_input,
        output=safe_output,
        error=item.error_message or "",
        approval_id=approval_id or metadata.get("approval_id"),
        created_at=item.created_at,
        completed_at=completed_at or metadata.get("completed_at") or (datetime.now(UTC).replace(tzinfo=None) if item.status in {"completed", "failed", "blocked", "waiting_approval"} else None),
    )
