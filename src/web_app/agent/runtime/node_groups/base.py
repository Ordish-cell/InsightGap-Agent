import json
import logging
import re
import time
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.llm.config import get_llm_settings
from src.web_app.agent.llm.errors import LLMInvocationError, LLMParseError, LLMUnavailableError
from src.web_app.agent.llm.factory import get_chat_model
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.agent.llm.usage import record_llm_call
from src.web_app.agent.runtime.checkpoint import record_event, record_step
from src.web_app.agent.runtime.intent_llm import infer_home_intent_with_llm, llm_select_tools
from src.web_app.agent.runtime.intent_schema import HomeIntentResult, LLMToolSelectionResult
from src.web_app.agent.runtime.langgraph_status import append_status_step
from src.web_app.agent.runtime.pipeline_steps import append_pipeline_step, ensure_pipeline_step
from src.web_app.agent.runtime.parallel_read import parallel_read_stage as run_parallel_read_stage
from src.web_app.agent.runtime.planner import plan_route
from src.web_app.agent.runtime.prefetch import parallel_prefetch as run_parallel_prefetch
from src.web_app.agent.runtime.router import route_user_input
from src.web_app.agent.runtime.schemas import (
    AgentResult,
    EvaluationResult,
    append_agent_result,
    execution_plan_from_route_plan,
    task_id_for_agent,
)
from src.web_app.agent.runtime.state import AgentRuntimeState, append_error, append_output, mark_completed
from src.web_app.agent.runtime.visible_thoughts import emit_visible_thought, visible_thought_texts
from src.web_app.mcp.registry import BUILTIN_TOOLS
from src.web_app.mcp.tool_router import _build_email_input, infer_tool, validate_tool_input
from src.web_app.context.builder import ContextBuilder
from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository, AgentStepRepository
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.artifact_repository import ArtifactRepository
from src.web_app.db.repositories.feed_repository import FeedRepository
from src.web_app.db.repositories.profile_repository import ProfileRepository
from src.web_app.research.schemas import ResearchRequest
from src.web_app.services.artifact_service import artifact_service
from src.web_app.services.memory_service import memory_service
from src.web_app.services.permission_service import PermissionGuard
from src.web_app.services.rag_service import rag_service
from src.web_app.services.research_service import research_service
from src.web_app.services.skill_service import skill_service
from src.web_app.services.mcp_service import mcp_service
from src.web_app.services.user_growth_service import user_growth_service

EXTERNAL_WRITE_TERMS = ("发邮件", "发送邮件", "邮件", "评论", "发布", "提交表单")
_EN_EXTERNAL_WRITE_KEYWORDS = ("email", "send", "post", "submit")
HIGH_RISK_TERMS = ("删除", "支付", "付款", "转账", "delete", "payment")

logger = logging.getLogger(__name__)


def _is_memory_like_input(user_text: str) -> bool:
    text = user_text.strip()
    memory_prefixes = (
        "以后", "从此", "从今", "记住", "帮我记", "记一下",
        "这个项目用", "这个项目是", "项目技术栈", "项目用",
        "默认用", "默认使用", "默认",
        "不要再", "别再给我", "别再", "不要给我",
        "我偏好", "我的偏好", "我喜欢", "我习惯",
        "我的项目", "我的技术栈", "我在用", "我用的",
        "长期", "永远", "一直",
    )
    for prefix in memory_prefixes:
        if text.startswith(prefix): return True
    return False

def _has_explicit_email_send_intent(user_text: str) -> bool:
    text = user_text.lower()
    has_send_keyword = any(kw in text for kw in (
        "发邮件", "发送邮件", "send email", "send mail", "寄邮件",
        "发给", "发一封给", "通知邮箱",
    )) or bool(re.search(r'\bsend\b.*\bemail\b', text)) or bool(re.search(r'\bemail\b.*\bsend\b', text))
    if not has_send_keyword: return False
    has_recipient = bool(re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', user_text)) or any(kw in user_text for kw in ("发给", "给…发", "给某某"))
    return has_recipient

def _is_obvious_email_intent(db: Session, user_text: str, route_plan: dict[str, Any]) -> bool:
    """Check if user clearly wants to send email — prevents false tool_not_found.

    Conditions: email.send is registered AND (user has email address in text OR clear email keywords).
    """
    from src.web_app.mcp.registry import registry as mcp_registry
    if mcp_registry.get_tool(db, "email.send") is None:
        return False
    text = user_text.lower()
    has_address = bool(re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', user_text))
    has_email_semantics = any(kw in text for kw in ("发邮件", "发送邮件", "send email", "send mail", "寄邮件", "发给", "发一封", "通知邮箱")) or (re.search(r'\b' + re.escape("email") + r'\b', text) is not None)
    return has_address or has_email_semantics


def _build_missing_fields_answer(
    tool_name: str,
    provided_args: dict[str, Any],
    missing_fields: list[dict[str, str]],
) -> str:
    """Generate a natural-language Chinese response asking the user to fill in missing fields.

    Does NOT create approval, does NOT execute the tool.
    """
    # ── Per-tool templates for what was recognized ──
    if tool_name == "email.send":
        parts = ["可以，我已经识别到你要发送邮件"]
        if provided_args.get("to"):
            parts.append(f"收件人是 {provided_args['to']}")
        if provided_args.get("subject"):
            parts.append(f"主题是「{provided_args['subject']}」")
        if provided_args.get("body"):
            body_preview = str(provided_args["body"])[:100]
            parts.append(f"正文是「{body_preview}」")
        questions = [m["question"] for m in missing_fields]
        answer = "，".join(parts) + "。还需要补充：" + "；".join(questions)
        return answer

    elif tool_name in ("local_file.write", "local_file.append"):
        parts = ["可以，我已经识别到你要写入文件"]
        if provided_args.get("path"):
            parts.append(f"路径是「{provided_args['path']}」")
        if provided_args.get("content"):
            content_preview = str(provided_args["content"])[:100]
            parts.append(f"内容是「{content_preview}」")
        questions = [m["question"] for m in missing_fields]
        answer = "，".join(parts) + "。还需要补充：" + "；".join(questions)
        return answer

    elif tool_name == "local_file.read":
        questions = [m["question"] for m in missing_fields]
        return "可以，但还需要补充信息：" + "；".join(questions)

    # ── Generic fallback ──
    questions = [m["question"] for m in missing_fields]
    provided_desc = ", ".join(f"{k}={str(v)[:80]}" for k, v in provided_args.items())
    return f"可以，我已经识别到你要执行 {tool_name}。已提供：{provided_desc}。还需要补充：" + "；".join(questions)


# ── Module-level helpers ─────────────────────────────────────────

_SENSITIVE_ARG_KEYS = {"password", "token", "secret", "api_key", "access_key", "smtp_password", "credential", "auth"}


def _approval_context_line(state: dict[str, Any]) -> str:
    """Return the approval-instruction line for the final LLM prompt.

    On resume (after approve/reject), the LLM must NOT say "需要审批".
    Instead it should describe what actually happened.
    """
    resume_context = str(state.get("_resume_context") or state.get("resume_token") or "")
    tool_name = state.get("pending_tool_name") or (state.get("tool_call") or {}).get("tool_name") or ""
    tool_status = (state.get("tool_call") or {}).get("status", "")
    tool_error = state.get("_tool_error") or (state.get("tool_result") or {}).get("error", "")

    if resume_context.startswith("rejected:") or tool_status == "rejected":
        return "用户已取消该操作，没有执行。请用自然语言说明已取消，并给出替代建议（如有）。不要声称已经执行。"
    if resume_context.startswith("approved:") and tool_status == "completed":
        return "该操作已获得用户批准并已执行成功。请基于工具执行结果回答，不要声称需要审批。"
    if resume_context.startswith("failed:") or tool_error:
        disp_name = _tool_display_name_local(tool_name)
        return f"该操作已获得用户批准但执行失败（{disp_name}: {tool_error[:200]}）。请用自然语言说明失败原因，不要声称已经执行成功，也不要声称需要审批。"
    if state.get("status") == "resuming":
        return "该操作已获得批准。请基于工具结果回答，不要声称需要审批。"
    # Normal path (no resume context)
    return "如果涉及 L3/L4 风险动作，说明需要审批，不能声称已经执行。"


def _tool_display_name_local(tool_name: str) -> str:
    names = {
        "email.send": "发送邮件",
        "local_file.write": "写入文件",
        "local_file.append": "追加文件",
        "local_file.delete": "删除文件",
        "local_file.read": "读取文件",
        "local_file.list": "列出文件",
        "web.search": "联网搜索",
        "system.time": "读取本地时间",
        "system.calc": "本地计算",
        "system.unit_convert": "本地单位换算",
        "system.uuid": "生成 UUID",
        "system.hash": "本地哈希计算",
    }
    return names.get(tool_name, tool_name)


def _sanitize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of tool_args with sensitive fields redacted."""
    if not args:
        return {}
    safe: dict[str, Any] = {}
    for key, value in args.items():
        lower_key = key.lower()
        if any(s in lower_key for s in _SENSITIVE_ARG_KEYS):
            safe[key] = "[redacted]"
        elif isinstance(value, str) and len(value) > 500:
            safe[key] = value[:500] + "..."
        else:
            safe[key] = value
    return safe


class BaseNodesMixin:
    def __init__(self, db: Session, payload: dict[str, Any], stream_queue: Any = None):
        self.db = db
        self.payload = payload
        self._stream_queue = stream_queue


__all__ = [name for name in globals() if not name.startswith("__")]
