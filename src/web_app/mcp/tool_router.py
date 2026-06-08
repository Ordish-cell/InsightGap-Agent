from typing import Any

TOOL_HINTS = {
    "search": "search_mcp.search",
    "github": "github_mcp.repo_summary",
    "artifact": "artifact_mcp.create_text_artifact",
    "memory": "memory_mcp.search",
    "skill": "skill_mcp.create_draft",
    "email": "email_mcp.create_draft",
    "email_send": "email.send",
    "browser": "browser_mcp.plan_actions",
    "local_file_list": "local_file.list",
    "local_file_read": "local_file.read",
    "local_file_write": "local_file.write",
    "local_file_append": "local_file.append",
    "local_file_delete": "local_file.delete",
}

# Map intent names to tool names
_INTENT_TOOL_MAP: dict[str, str] = {
    "tool.email": "email.send",
    "tool.local_file": "local_file.write",
    "tool.shell_readonly": "local_file.list",
    "tool.shell_write": "local_file.write",
    "tool.dangerous": "local_file.delete",
    "tool.browser": "browser_mcp.plan_actions",
    "tool.comment": "email.send",
    "tool.form_submit": "email.send",
}


def infer_tool(user_input: str, payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if payload.get("tool_name"):
        return payload["tool_name"], payload.get("tool_input", payload.get("input", {}))
    if payload.get("intent"):
        tool_name = _INTENT_TOOL_MAP.get(payload["intent"])
        if tool_name:
            return tool_name, _build_input_for_tool(tool_name, user_input, payload)

    text = user_input.lower()
    if "github" in text:
        return TOOL_HINTS["github"], {"repo": payload.get("repo", "owner/name")}

    # ── Email ───────────────────────────────────────────────
    _send_triggers = ("发邮件", "发送邮件", "发一封", "send email", "send mail")
    if any(t in text for t in _send_triggers) or ("email" in text and "draft" not in text):
        return "email.send", _build_email_input(user_input, payload)

    if "browser" in text or "浏览器" in user_input:
        return TOOL_HINTS["browser"], {"goal": user_input, "url": payload.get("url", "")}

    # ── Local file ──────────────────────────────────────────
    if any(t in text for t in ("创建文件", "写入文件", "写文件", "保存到文件", "新建文件", "写一个")):
        return "local_file.write", _build_file_write_input(user_input, payload)
    if any(t in text for t in ("读取文件", "查看文件", "打开文件", "看看文件")):
        return "local_file.read", {"path": payload.get("path", ""), "max_chars": payload.get("max_chars")}
    if any(t in text for t in ("列出目录", "列出文件", "看看本地", "查看目录", "看看workspace")):
        return "local_file.list", {"path": payload.get("path", ".")}
    if any(t in text for t in ("删除文件",)):
        return "local_file.delete", {"path": payload.get("path", "")}
    if any(t in text for t in ("追加", "追加写入", "append")):
        return "local_file.append", {"path": payload.get("path", ""), "content": payload.get("content", user_input)}

    return None, {}


def _build_email_input(user_input: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "to": payload.get("to", ""),
        "subject": payload.get("subject", user_input[:80]),
        "body": payload.get("body", user_input),
    }


def _build_file_write_input(user_input: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": payload.get("path", ""),
        "content": payload.get("content", user_input),
        "mode": payload.get("mode", "create_or_overwrite"),
    }


def _build_input_for_tool(tool_name: str, user_input: str, payload: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "email.send":
        return _build_email_input(user_input, payload)
    if tool_name == "local_file.write":
        return _build_file_write_input(user_input, payload)
    if tool_name in ("local_file.list",):
        return {"path": payload.get("path", ".")}
    if tool_name in ("local_file.read",):
        return {"path": payload.get("path", ""), "max_chars": payload.get("max_chars")}
    if tool_name == "local_file.append":
        return {"path": payload.get("path", ""), "content": payload.get("content", user_input)}
    return {}
