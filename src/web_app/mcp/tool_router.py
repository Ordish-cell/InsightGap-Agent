from typing import Any

TOOL_HINTS = {
    "search": "search_mcp.search",
    "github": "github_mcp.repo_summary",
    "artifact": "artifact_mcp.create_text_artifact",
    "memory": "memory_mcp.search",
    "skill": "skill_mcp.create_draft",
    "email": "email_mcp.create_draft",
    "browser": "browser_mcp.plan_actions",
}


def infer_tool(user_input: str, payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if payload.get("tool_name"):
        return payload["tool_name"], payload.get("tool_input", payload.get("input", {}))
    text = user_input.lower()
    if "github" in text:
        return TOOL_HINTS["github"], {"repo": payload.get("repo", "owner/name")}
    if "email" in text or "邮件" in user_input:
        return TOOL_HINTS["email"], {"subject": payload.get("subject", user_input[:80]), "body": payload.get("body", user_input)}
    if "browser" in text or "浏览器" in user_input:
        return TOOL_HINTS["browser"], {"goal": user_input, "url": payload.get("url", "")}
    return None, {}
