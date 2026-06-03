from src.web_app.core.constants import BLOCKED_TOOL_PATTERNS, L3_EXTERNAL_WRITE, L4_HIGH_RISK


class PermissionGuard:
    def check_tool_call(self, tool_name: str, permission_level: str, approval_required: bool = False) -> dict[str, str | bool]:
        normalized = tool_name.lower()
        if permission_level == L4_HIGH_RISK:
            return {"allowed": False, "requires_approval": False, "reason": "high_risk_denied"}
        if permission_level == L3_EXTERNAL_WRITE or approval_required or any(pattern in normalized for pattern in BLOCKED_TOOL_PATTERNS):
            return {"allowed": False, "requires_approval": True, "reason": "approval_required"}
        return {"allowed": True, "requires_approval": False, "reason": "allowed"}
