from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.web_app.agent.runtime.intent_schema import LLMToolSelectionResult

logger = logging.getLogger(__name__)

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


def infer_tool(
    user_input: str,
    payload: dict[str, Any],
    llm_result: "LLMToolSelectionResult | None" = None,  # noqa: F821
) -> tuple[str | None, dict[str, Any]]:
    """Determine tool name and input from user text, LLM result, or payload.

    Priority:
    1. Explicit tool_name in payload
    2. LLM result (confidence >= 0.5)
    3. Intent → tool map from payload
    4. Keyword matching on user_input (fallback)
    """
    # 1. Explicit tool_name
    if payload.get("tool_name"):
        return payload["tool_name"], payload.get("tool_input", payload.get("input", {}))

    # 2. LLM result (primary path for natural language)
    if llm_result is not None and llm_result.confidence >= 0.5 and llm_result.tool_calls:
        from src.web_app.mcp.registry import normalize_tool_name
        first_call = llm_result.tool_calls[0]
        raw_name = first_call.name
        canonical = normalize_tool_name(raw_name)
        args = first_call.arguments
        cleaned_args, missing = validate_tool_input(canonical, args)
        logger.info(
            "[LLM_TOOL_SELECT_DEBUG] available_tools_count=%d user_text=%.200s raw_model_output=%.300s "
            "parsed_tool_calls=[%s] normalized_tool=%s final_args=%s missing_fields=%s confidence=%.2f",
            -1,  # count filled in caller
            user_input[:200],
            raw_name[:300],
            canonical,
            canonical,
            str(cleaned_args)[:300],
            str(missing)[:200],
            llm_result.confidence,
        )
        return canonical, cleaned_args

    # 3. Intent → tool map
    if payload.get("intent"):
        tool_name = _INTENT_TOOL_MAP.get(payload["intent"])
        if tool_name:
            return tool_name, _build_input_for_tool(tool_name, user_input, payload)

    # 4. Keyword fallback (existing behavior, unchanged)
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


def _parse_email_fields(text: str) -> dict[str, str]:
    """Extract to/subject/body from Chinese email request text.

    Supports:
      - 发邮件给 test@example.com，主题 Hello，正文 This is a test
      - 给 test@example.com 发邮件，主题是 Hello，正文是 This is a test
      - 收件人 test@example.com 主题 Hello 正文 This is a test
    """
    import re

    to = ""
    subject = ""
    body = ""

    # ── Extract to (email address) ──────────────────────────
    to_match = re.search(
        r'(?:发给|发邮件给|发送给|给|收件人[：:\s]*)\s*'
        r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        text,
    )
    if to_match:
        to = to_match.group(1).strip()

    # Also try bare email in the text (发邮件给 xxx@yy.com)
    if not to:
        bare_email = re.search(
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            text,
        )
        if bare_email:
            to = bare_email.group(1).strip()

    # ── Extract subject ──────────────────────────────────────
    # Match: 主题[：:是]? <text> (stop before 正文/内容/body/.)
    subj_match = re.search(
        r'主题[：:\s]*(?:是\s*)?(.+?)(?=\s*(?:正文|内容|body|。|$))',
        text,
    )
    if subj_match:
        subject = subj_match.group(1).strip().rstrip('，,')
        # Truncate at "正文" or "内容" marker if present
        for marker in ('正文', '内容', 'body'):
            idx = subject.find(marker)
            if idx >= 0:
                subject = subject[:idx].strip().rstrip('，,')

    # ── Extract body ─────────────────────────────────────────
    # Match: 正文[：:是]? <text> (to end or next known marker)
    body_match = re.search(
        r'(?:正文|内容|body)[：:\s]*(?:是\s*)?(.+)',
        text,
        re.IGNORECASE,
    )
    if body_match:
        body = body_match.group(1).strip()

    # If subject is still empty but there's text between to-email and body,
    # extract it
    if not subject and to:
        # Find text between email extract and "正文" marker
        after_to = text.split(to, 1)[-1] if to in text else text
        # Remove leading punctuation
        after_to = re.sub(r'^[，,\s]*', '', after_to)
        # Find subject-like patterns
        subj_match2 = re.search(
            r'(?:主题[：:\s]*(?:是\s*)?)?(.+?)(?=\s*(?:正文|内容|body|。|$))',
            after_to,
        )
        if subj_match2:
            subject = subj_match2.group(1).strip().rstrip('，,')
            # If the extracted text is just the email address again or empty, clear it
            if '@' in subject and len(subject) < 50:
                subject = ""

    return {"to": to, "subject": subject, "body": body}


def _build_email_input(user_input: str, payload: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_email_fields(user_input)
    return {
        "to": payload.get("to") or parsed["to"],
        "subject": payload.get("subject") or parsed["subject"] or "",
        "body": payload.get("body") or parsed["body"] or user_input,
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


# ── Tool input validation ─────────────────────────────────────────

def validate_tool_input(
    tool_name: str, arguments: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate arguments against the tool's input_schema.

    Returns (cleaned_args, missing_fields).
    missing_fields items: {"field": "...", "question": "..."}
    """
    from src.web_app.mcp.registry import BUILTIN_TOOLS

    spec = None
    for t in BUILTIN_TOOLS:
        if t.name == tool_name:
            spec = t
            break

    if spec is None:
        logger.warning("[TOOL_NOT_FOUND_DEBUG] tool=%s not in BUILTIN_TOOLS", tool_name)
        return arguments, []

    schema = spec.input_schema or {}
    properties = schema.get("properties", {})
    required: list[str] = schema.get("required", []) if isinstance(schema.get("required"), list) else []

    cleaned: dict[str, Any] = {}
    missing: list[dict[str, str]] = []

    # ── Field question templates ──
    _FIELD_QUESTIONS_ZH: dict[str, dict[str, str]] = {
        "email.send": {
            "to": "收件人邮箱是什么？",
            "subject": "邮件主题是什么？",
            "body": "邮件正文是什么？",
        },
        "local_file.write": {
            "path": "文件路径是什么？",
            "content": "文件内容是什么？",
        },
        "local_file.read": {
            "path": "要读取哪个文件？",
        },
        "local_file.append": {
            "path": "要追加到哪个文件？",
            "content": "要追加什么内容？",
        },
        "local_file.delete": {
            "path": "要删除哪个文件？",
        },
        "local_file.list": {
            "path": "要列出哪个目录的文件？",
        },
    }
    field_questions = _FIELD_QUESTIONS_ZH.get(tool_name, {})

    for field_name in required:
        raw = arguments.get(field_name)
        present = raw is not None and (not isinstance(raw, str) or raw.strip() != "")
        if present:
            cleaned[field_name] = raw.strip() if isinstance(raw, str) else raw
        else:
            question = field_questions.get(field_name, f"请提供 {field_name}")
            missing.append({"field": field_name, "question": question})

    # Copy optional fields too
    for field_name, field_schema in properties.items():
        if field_name not in required and field_name in arguments and arguments[field_name]:
            cleaned[field_name] = arguments[field_name]

    # ── email.send: basic format check on `to` ──
    if tool_name == "email.send" and "to" in cleaned:
        import re
        to_val = cleaned["to"]
        if not re.search(r'@.{2,}', to_val):
            cleaned.pop("to", None)
            missing.append({
                "field": "to",
                "question": f"收件人地址 \"{to_val[:50]}\" 格式不对，请提供正确的邮箱地址。",
            })

    if missing:
        logger.debug(
            "[LLM_TOOL_SELECT_DEBUG] tool=%s missing_fields=%s args=%s",
            tool_name, str(missing), str(cleaned)[:200],
        )

    return cleaned, missing
