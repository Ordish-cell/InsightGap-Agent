"""Conservative first-person facts and turn-scoped consent; no model inference."""
import re
from contextvars import ContextVar

LABELS = {"preferred_name": "称呼", "response_language": "回答语言", "script_preference": "简繁偏好"}
EXPLICIT_CONFIRMATIONS = {"确认记住", "确认保存", "请记住", "保存这些记忆"}
SHORT_CONFIRMATIONS = {"好", "好的", "是", "可以", "同意", "确认"}

# Only the save hook sets this context, never tool arguments or model output.
authorized_fact = ContextVar("authorized_basic_fact", default=None)


def parse_facts(text):
    text = text.strip()
    if re.search(r'[\n\r>"“”「」『』`?？]|假如|假设|如果|比如|例如|这次|本次|暂时|不要|别记|不保存', text):
        return [], False
    explicit = bool(re.match(r"^(?:请)?(?:记住|记一下|记下来)[，,:：\s]*", text))
    text = re.sub(r"^(?:请)?(?:记住|记一下|记下来)[，,:：\s]*", "", text)
    facts = {}
    for part in re.split(r"[，,。；;！!]", text):
        part = part.strip()
        if not part:
            continue
        name = re.fullmatch(r"(?:我叫|以后(?:请)?叫我)([\u4e00-\u9fff]{1,4}|[A-Za-z][A-Za-z -]{0,29})", part)
        language = re.fullmatch(r"(?:以后|今后)(?:请)?用(中文|英文|英语|日语|日文|韩语|法语|德语)(?:回答|回复)(?:我)?", part)
        script = re.fullmatch(r"(?:以后|今后)(?:请)?用(简体|繁体)(?:字|中文)?(?:回答|回复)?", part)
        if name:
            key, value = "preferred_name", name[1]
            if re.search(r"^(?:你|他|她|它|大家)|什么|谁|吗|呢|不是|不要|以后|回答|记住|的话|就|过来|过去", value):
                return [], False
        elif language:
            key, value = "response_language", {"英文": "英语", "日文": "日语"}.get(language[1], language[1])
        elif script:
            key, value = "script_preference", script[1]
        else:
            return [], False
        if key in facts and facts[key] != value:
            return [], False
        facts[key] = value
    return [{"key": k, "value": v} for k, v in facts.items()], explicit


def content_for(fact):
    return f"用户确认的{LABELS[fact['key']]}：{fact['value']}"


def prepare(db, state):
    from src.web_app.db.repositories.agent_repository import AgentChatMessageRepository
    request = state.get("request", {})
    explicit_fields = request.get("_explicit_fields", request.keys())
    blocked = ("write_memory" in explicit_fields and request.get("write_memory") is False) or bool(
        re.search(r"不要记住|不用记住|别记住|不要保存|不保存", state["user_input"]))
    facts, direct = parse_facts(state["user_input"])
    plan = {"facts": facts, "authorized": direct and not blocked, "blocked": bool(blocked),
            "source_run_id": state["run_id"], "confirmation": "explicit_request" if direct else "pending"}
    text = state["user_input"].strip().rstrip("。！.! ")
    if not facts and not blocked and db is not None and text in EXPLICIT_CONFIRMATIONS | SHORT_CONFIRMATIONS:
        rows = AgentChatMessageRepository(db).list_recent_by_conversation(state["user_id"], state["conversation_id"], limit=8)
        rows = [m for m in rows if m.run_id != state["run_id"]]
        # Require the immediately preceding message, not an older proposal behind a refusal/topic change.
        last = rows[-1] if rows else None
        if last and last.role == "assistant" and last.status == "completed":
            final = (last.metadata_json or {}).get("final_response") or {}
            proposal = final.get("memory_proposal")
            explicit_yes = text in EXPLICIT_CONFIRMATIONS
            short_yes = text in SHORT_CONFIRMATIONS
            ambiguous = final.get("research_proposal") or final.get("approval_required") or (last.metadata_json or {}).get("research_proposal")
            if proposal and (explicit_yes or (short_yes and not ambiguous)):
                # Reparse the original user statement; never trust arbitrary JSON facts.
                from src.web_app.models.orm import AgentRun
                source = db.get(AgentRun, proposal.get("source_run_id"))
                if source and source.id == last.run_id and source.status == "completed" and source.user_id == state["user_id"] and source.conversation_id == state["conversation_id"]:
                    parsed, _ = parse_facts(source.user_input)
                    if parsed and parsed == proposal.get("facts"):
                        plan.update(facts=parsed, authorized=True, source_run_id=source.id,
                                    proposal_message_id=last.id, confirmation="previous_proposal")
    state["basic_memory"] = plan
    return plan
