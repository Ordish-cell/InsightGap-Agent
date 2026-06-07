from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.agent.runtime.visibility import should_show_visible_thought_to_user


def build_visible_thought_step(step: str | dict[str, Any], state: dict[str, Any]) -> str:
    key = _step_key(step)
    route_plan = state.get("route_plan") or {}
    home_intent = state.get("home_intent") or {}
    risk_level = str(route_plan.get("risk_level") or home_intent.get("risk_level") or "L0")
    intent = str(route_plan.get("intent") or home_intent.get("intent") or home_intent.get("detected_intent") or state.get("route") or "chat")
    needs_approval = bool(state.get("approval_required") or route_plan.get("needs_approval") or home_intent.get("needs_approval"))

    if key in {"permission_guard", "home_intent_react", "home_intent"}:
        if needs_approval:
            return f"我先判断这个请求的类型和风险等级；它属于 {risk_level} 风险，继续前需要先走审批保护。"
        return f"我先判断这个请求属于{_intent_label(intent)}，风险等级是 {risk_level}，确认可以继续处理。"
    if key == "planner":
        return "我会把问题拆成几个可处理的小任务，先确定回答路线，再组织结果。"
    if key == "context_builder":
        return "我正在整理当前会话上下文，比如用户问题、已有偏好、时间地点等信息，避免回答跑偏。"
    if key == "skill_matcher":
        if state.get("matched_skill"):
            return "我发现有可复用的技能或流程，可以借用已有经验加速处理。"
        return "我没有发现完全匹配的现成技能，所以会按当前问题重新组织答案。"
    if key in {"research_agent", "research"}:
        return "我开始检索和整合相关信息，优先保留能直接帮助用户决策的内容。"
    if key in {"rag_agent", "rag"}:
        return "我正在检索知识库里的相关内容，并把能支持回答的依据先筛出来。"
    if key in {"tool_agent", "tool"}:
        if needs_approval or state.get("status") == "waiting_approval":
            return "我已经识别到可能涉及外部动作，会先停在审批环节，避免直接执行高风险操作。"
        return "我正在确认是否需要调用工具，并只保留和这次请求直接相关的执行结果。"
    if key in {"artifact_agent", "artifact"}:
        return "我会把前面的结果整理成可保存、可复用的产物，而不是只停留在零散说明。"
    if key in {"memory_agent", "memory_writer"}:
        return "我正在判断这次对话里是否有值得保存的偏好或结论，避免把闲聊误写成长期记忆。"
    if key in {"skill_agent", "skill_draft_detector", "skill_librarian"}:
        return "我正在判断这次流程是否值得沉淀为可复用技能，只有明显可复用时才会生成草稿。"
    if key == "evaluator":
        return "我正在检查前面的结果是否完整、是否有明显风险或遗漏。"
    if key == "final_response":
        return "我已经整理好结论，下面给出可以直接使用的回答。"
    return "我已经完成这一阶段的处理，并把结果交给下一步继续整理。"


def emit_visible_thought(db: Session, state: dict[str, Any], step: str | dict[str, Any], *, status: str = "completed") -> str:
    text = build_visible_thought_step(step, state).strip()
    if not text:
        return ""

    key = _step_key(step)
    thoughts = list(state.get("visible_thoughts") or [])
    if any(_thought_key(item) == key for item in thoughts):
        return text

    entry = {
        "key": key,
        "id": f"thought-{len(thoughts) + 1:03d}",
        "text": text,
        "status": status,
        "visibility": "user",
        "source": "visible_thought",
        "created_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }
    thoughts.append(entry)
    state["visible_thoughts"] = thoughts

    langgraphstatus = state.setdefault("langgraphstatus", {})
    langgraphstatus["visible_thoughts"] = thoughts

    record_event(
        db,
        state["run_id"],
        "visible_thought",
        {"text": text, "status": status},
        node_name=key,
        user_id=state.get("user_id"),
        thread_id=state.get("thread_id", ""),
    )
    stream_queue = state.get("_stream_queue")
    if stream_queue and should_show_visible_thought_to_user(state, key):
        base_payload = {name: entry[name] for name in ("id", "status", "visibility", "source", "created_at")}
        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]
        for index, sentence in enumerate(sentences, start=1):
            stream_queue.put_nowait(
                {
                    "event": "visible_thought_delta",
                    "data": {
                        "run_id": state.get("run_id"),
                        "thread_id": state.get("thread_id", ""),
                        "event_type": "visible_thought_delta",
                        "node_name": key,
                        "visibility": "user",
                        "display_channel": "thinking",
                        "payload": {**base_payload, "text": sentence, "index": index, "status": "streaming"},
                        "created_at": entry["created_at"],
                    },
                }
            )
        stream_queue.put_nowait(
            {
                "event": "visible_thought_delta",
                "data": {
                    "run_id": state.get("run_id"),
                    "thread_id": state.get("thread_id", ""),
                    "event_type": "visible_thought_delta",
                    "node_name": key,
                    "visibility": "user",
                    "display_channel": "thinking",
                    "payload": {**base_payload, "text": "", "full_text": text, "status": status},
                    "created_at": entry["created_at"],
                },
            }
        )
    return text


def _split_sentences(text: str, max_len: int = 80) -> list[str]:
    """Split text by sentence boundaries (。！？!?；;\\n), keeping punctuation at sentence ends."""
    result: list[str] = []
    current = ""
    boundaries = set("。！？；!?;\n")
    for char in text:
        current += char
        if char in boundaries or len(current) >= max_len:
            stripped = current.strip()
            if stripped:
                result.append(stripped)
            current = ""
    remaining = current.strip()
    if remaining:
        result.append(remaining)
    return result


def visible_thought_texts(state: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in state.get("visible_thoughts") or []:
        text = item.get("text") if isinstance(item, dict) else item
        text = str(text or "").strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def _step_key(step: str | dict[str, Any]) -> str:
    if isinstance(step, dict):
        return str(step.get("key") or step.get("node_name") or step.get("name") or "").strip()
    return str(step or "").strip()


def _thought_key(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("key") or "").strip()
    return ""


def _intent_label(intent: str) -> str:
    labels = {
        "chat": "轻量对话请求",
        "research": "研究任务",
        "rag": "知识库问答",
        "artifact": "产物生成任务",
        "tool": "工具相关任务",
        "tool.email": "邮件工具任务",
        "tool.browser": "浏览器工具任务",
        "tool.comment": "评论发布任务",
        "tool.form_submit": "表单提交任务",
        "memory": "记忆整理任务",
        "skill": "技能沉淀任务",
        "feed_research": "信息卡片研究任务",
        "mixed": "复合任务",
    }
    return labels.get(intent, "当前请求")
