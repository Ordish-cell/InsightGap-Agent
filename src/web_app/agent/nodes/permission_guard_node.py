from src.web_app.agent.state import AgentState
from src.web_app.services.permission_service import PermissionGuard


def permission_guard_node(state: AgentState) -> AgentState:
    guard = PermissionGuard()
    approvals = []
    for call in state.get("tool_calls", []):
        decision = guard.check_tool_call(call.get("tool_name", ""), call.get("permission_level", "L0_READ_ONLY"), call.get("approval_required", False))
        if decision["requires_approval"]:
            approvals.append({"tool_call": call, "decision": decision})
        if not decision["allowed"] and not decision["requires_approval"]:
            state["error"] = decision["reason"]
    state["approvals"] = approvals
    return state
