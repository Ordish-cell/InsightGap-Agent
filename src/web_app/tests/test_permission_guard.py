from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.services.permission_service import PermissionGuard


def test_permission_guard_l3_l4():
    guard = PermissionGuard()
    assert guard.check_tool_call("email/send", L3_EXTERNAL_WRITE)["requires_approval"] is True
    assert guard.check_tool_call("delete", L4_HIGH_RISK)["allowed"] is False
