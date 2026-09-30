"""Server-owned scope, consent and recall rules shared by every memory reader."""
import re
import math
from datetime import UTC, datetime

from sqlalchemy import select

from src.web_app.models.orm import AgentConversation, AgentRun, UserProfile

PERSONAL_CATEGORIES = {
    "preferred_name", "response_language", "script_preference", "name_preference",
    "language_preference", "tone_preference", "answer_preference", "output_preference",
}
DENIAL = re.compile(r"不要记住|不用记住|别记住|不要保存|不保存|不记录|(?:do not|don't)\s+(?:remember|save|memorize)", re.I)


def settings_for(db, user_id, conversation_id=None, request=None):
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    conversation = db.scalar(select(AgentConversation).where(
        AgentConversation.user_id == user_id, AgentConversation.conversation_id == conversation_id)) if conversation_id else None
    meta = (conversation.metadata_json or {}) if conversation else {}
    overrides = meta.get("memory_settings") or {}
    request = request or {}
    explicit = request.get("_explicit_fields", request.keys())
    result = {
        "use_memory": profile.use_memory if profile else True,
        "generate_memory": profile.generate_memory if profile else False,
        "profile_version": profile.memory_settings_version if profile else 0,
        "conversation_version": meta.get("memory_settings_version", 0),
        "conversation_generation": meta.get("context_generation", 0),
        "overrides": dict(overrides),
    }
    for key in ("use_memory", "generate_memory"):
        if isinstance(overrides.get(key), bool):
            result[key] = overrides[key]
        if key in explicit and isinstance(request.get(key), bool):
            result[key] = request[key]
    result["writes_blocked"] = (
        "write_memory" in explicit and request.get("write_memory") is False
    ) or bool(DENIAL.search(str(request.get("user_input") or request.get("input") or "")))
    if result["writes_blocked"]:
        result["generate_memory"] = False
    return result


def runtime_policy(db):
    from src.web_app.agent.runtime.chat_control import execution
    token = execution.get()
    run = db.get(AgentRun, token.run_id) if token else None
    policy = dict(run.memory_policy or {}) if run else {}
    if run:
        conversation = db.scalar(select(AgentConversation).where(AgentConversation.user_id == run.user_id,
            AgentConversation.conversation_id == run.conversation_id))
        if conversation and (conversation.metadata_json or {}).get("context_generation", 0) != policy.get("conversation_generation", 0):
            policy.update(writes_blocked=True, use_memory=False)
    return policy


def validate_source(db, user_id, metadata):
    """Resolve provenance from owned user messages, never from model assertions."""
    from src.web_app.models.orm import AgentChatMessage
    from src.web_app.agent.runtime.chat_control import execution
    token = execution.get()
    run_id = (token.run_id if token else None) or metadata.get("run_id") or metadata.get("agent_run_id")
    if not run_id:
        return metadata
    if not str(run_id).isdigit():
        return {**metadata, "status": "pending", "source_verified": False}
    run = db.get(AgentRun, int(run_id))
    if not run or run.user_id != user_id:
        raise ValueError("invalid_memory_source")
    source = db.scalar(select(AgentChatMessage).where(AgentChatMessage.user_id == user_id,
        AgentChatMessage.run_id == run.id, AgentChatMessage.conversation_id == run.conversation_id,
        AgentChatMessage.role == "user").order_by(AgentChatMessage.id).limit(1))
    quote = metadata.get("source_quote") or (source.content if source else "")
    verified = bool(source and quote and quote in source.content)
    return {**metadata, "run_id": run.id, "conversation_id": run.conversation_id,
            "source_message_id": source.id if source else None, "source_quote": quote,
            "source_verified": verified, **({"status": "pending"} if not verified else {})}


def infer_scope(metadata, conversation_id=""):
    # A tool-supplied scope is never an authorization to make project facts global.
    quote = str(metadata.get("source_quote") or "")
    contextual = bool(re.search(r"本项目|这个项目|当前项目|本次|这次|本轮|暂时|this (?:project|task|time)|for now", quote, re.I))
    # A model labeling a technical fact as a personal category cannot globalize it.
    habitual = bool(re.search(r"我(?:一直|通常|长期|平时|总是|更|不)?(?:喜欢|偏好|习惯)|\bI\s+(?:(?:always|usually)\s+)?(?:prefer|like|dislike)\b", quote, re.I))
    lasting = bool(re.search(r"以后|一直|长期|习惯|通常|总是|今后|每次|记住|always|usually|from now on|remember", quote, re.I))
    personal_statement = not metadata.get("run_id") or (
        metadata.get("category") in {"preferred_name", "name_preference"}
        and bool(re.search(r"我叫|我的名字|叫我|call me|my name", quote, re.I))
    ) or (bool(re.search(r"回答|回复|解释|沟通|语言|简体|繁体|answer|respond|reply|explain|language|concise|tone", quote, re.I)) and (lasting or habitual))
    if not contextual and ((metadata.get("category") in PERSONAL_CATEGORIES and personal_statement) or (
        metadata.get("category") in {"preference", "negative_preference"}
        and metadata.get("personal_long_term") is True and metadata.get("source_verified") is True
        and habitual
    )):
        return "user", ""
    return ("conversation", str(conversation_id)) if conversation_id else ("legacy_unscoped", "")


def eligible(item, conversation_id=None, *, allow_legacy=False):
    meta = item.get("metadata") or {}
    if meta.get("status", "active") != "active" or not meta.get("visible_in_long_term_memory", True) or meta.get("sensitive"):
        return False
    try:
        confidence = float(meta.get("confidence", 1) or 0)
    except (ValueError, TypeError):
        return False
    if item.get("memory_type") not in {"semantic", "episodic"} or not math.isfinite(confidence) or confidence < .55:
        return False
    expiry = item.get("expires_at")
    if expiry:
        try:
            expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00")) if isinstance(expiry, str) else expiry
            expiry = expiry.replace(tzinfo=UTC) if expiry.tzinfo is None else expiry.astimezone(UTC)
            if expiry <= datetime.now(UTC):
                return False
        except (TypeError, ValueError):
            return False
    scope = item.get("scope", "legacy_unscoped")
    if scope == "legacy_unscoped" and meta.get("category") in PERSONAL_CATEGORIES:
        scope = "user"  # deterministic compatibility for unmigrated fixture records
    return scope == "user" or (scope == "conversation" and bool(conversation_id) and item.get("scope_id") == conversation_id) or (allow_legacy and scope == "legacy_unscoped")
