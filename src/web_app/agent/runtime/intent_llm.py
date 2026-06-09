import json
import time
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.llm.errors import LLMInvocationError, LLMParseError, LLMUnavailableError
from src.web_app.agent.llm.factory import get_chat_model
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.agent.llm.usage import record_llm_call
from src.web_app.agent.runtime.intent_schema import HomeIntentResult


def infer_home_intent_with_llm(
    db: Session,
    *,
    run_id: int,
    thread_id: str,
    user_id: int,
    user_input: str,
    page_context: dict[str, Any],
    selected_feed_card_id: Any = None,
    memory_summary: str = "",
) -> HomeIntentResult:
    resolution = resolve_model_name("intent", complexity="low")
    started = time.perf_counter()
    output_text = ""
    try:
        model = get_chat_model("intent", complexity="low", temperature=0)
        prompt = _build_prompt(
            user_input=user_input,
            page_context=page_context,
            selected_feed_card_id=selected_feed_card_id,
            memory_summary=memory_summary,
        )
        message = model.invoke(prompt)
        output_text = _message_content(message)
        payload = _parse_json(output_text)
        result = HomeIntentResult.model_validate(payload)
        result.model_used = resolution.model
        result.raw_intent_source = "llm"
        latency_ms = int((time.perf_counter() - started) * 1000)
        record_llm_call(
            db,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            node_name="home_intent_react",
            purpose="intent",
            provider=resolution.provider,
            model=resolution.model,
            tier=resolution.tier,
            latency_ms=latency_ms,
            status="completed",
            estimated_input_chars=len(prompt),
            estimated_output_chars=len(output_text),
            metadata={"input_preview": user_input[:200]},
        )
        return result
    except LLMUnavailableError as exc:
        _record_failed_call(db, run_id, thread_id, user_id, resolution, started, user_input, output_text, str(exc))
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        _record_failed_call(db, run_id, thread_id, user_id, resolution, started, user_input, output_text, str(exc))
        raise LLMParseError(str(exc)) from exc
    except Exception as exc:
        _record_failed_call(db, run_id, thread_id, user_id, resolution, started, user_input, output_text, str(exc))
        raise LLMInvocationError(str(exc)) from exc


def _build_prompt(user_input: str, page_context: dict[str, Any], selected_feed_card_id: Any, memory_summary: str) -> str:
    payload = {
        "user_input": user_input,
        "page_context": page_context,
        "selected_feed_card_id": selected_feed_card_id,
        "memory_summary": memory_summary,
        "available_capabilities": [
            "context_builder",
            "skill_matcher",
            "research_agent",
            "rag_agent",
            "artifact_agent",
            "tool_agent",
            "memory_agent",
            "skill_agent",
            "evaluator",
            "final_response",
        ],
        "current_route_options": ["chat", "research", "rag", "artifact", "feed_research", "tool", "memory", "skill", "mixed"],
    }
    return (
        "你是 Agent OS 的首页意图判断器。你的任务不是执行用户请求，而是判断用户想让 Agent 做什么，并输出严格 JSON。\n"
        "必须判断 intent、confidence、risk_level、needs_approval、needs_clarification、required_agents、expected_output、reason_summary、suggested_route_hints、tool_action_type。\n"
        "风险规则：L0=闲聊/解释；L1=搜索/研究/RAG/读取信息；L2=生成本地 artifact；L3=外部写入如发邮件、评论、发布、提交表单；L4=删除、支付、转账、权限修改、批量不可逆操作、安全配置变更。\n"
        "严格要求：只输出 JSON，不要 Markdown，不要代码块，不要 chain-of-thought，不要执行工具。required_agents 只能从输入的 available_capabilities 中选择。\n"
        "如果不确定，needs_clarification=true。不要把高风险动作标为低风险。\n"
        "输出字段：intent, confidence, risk_level, needs_approval, needs_clarification, required_agents, expected_output, reason_summary, suggested_route_hints, tool_action_type。\n"
        f"输入：{json.dumps(payload, ensure_ascii=False)}"
    )


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("Intent LLM output must be a JSON object")
    return data


def _record_failed_call(db: Session, run_id: int, thread_id: str, user_id: int, resolution, started: float, user_input: str, output_text: str, error: str) -> None:
    record_llm_call(
        db,
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        node_name="home_intent_react",
        purpose="intent",
        provider=resolution.provider,
        model=resolution.model,
        tier=resolution.tier,
        latency_ms=int((time.perf_counter() - started) * 1000),
        status="failed",
        error_message=error,
        estimated_input_chars=len(user_input),
        estimated_output_chars=len(output_text),
        metadata={"input_preview": user_input[:200]},
    )


# ── LLM-based tool selection ──────────────────────────────────────

def _build_tool_selection_prompt(
    user_input: str,
    available_tools: list[dict[str, Any]],
) -> str:
    """Build prompt for the tool-selection LLM call."""
    tools_json = json.dumps(available_tools, ensure_ascii=False, indent=2)
    return (
        "你是 Agent OS 的工具选择器，不是最终回答助手。\n\n"
        "你的任务：根据用户自然语言，判断是否需要调用工具，并输出严格 JSON。\n\n"
        "你必须遵守：\n"
        "1. 只输出 JSON，不要输出 Markdown，不要解释，不要代码块。\n"
        "2. 如果用户只是普通聊天、解释、研究、问答，不要强行选择工具。\n"
        "3. 如果用户明确要求执行外部动作（发送邮件、写文件、追加文件、提交表单、发表评论），应该选择对应工具。\n"
        "4. 工具名必须来自 available_tools 中的 canonical name，不能编造。\n"
        "5. 如果用户表达使用了别名、中文说法或自然语言说法，你必须映射到 canonical name。\n"
        "6. 对于发邮件请求，只要用户表达了\"发邮件 / 发送邮件 / email / mail / 通知某邮箱\"，"
        "并且 available_tools 存在 email.send，就必须选择 email.send。\n"
        "7. 不要因为句式不同而返回 route=unknown_tool。\n"
        "8. 如果工具存在但参数不完整，输出 missing_fields，不要返回 unknown_tool。\n"
        "9. 如果用户已经给出收件人、主题、正文，必须提取到 arguments。\n"
        "10. 不要编造没有出现的收件人、主题、正文。\n"
        "11. 对 L3/L4 工具，你只负责选择工具和参数，不要执行，不要跳过审批。\n"
        "12. 如果没有合适工具，返回 route=\"chat\" 或 route=\"unknown_tool\"，并说明 requested_action。\n"
        "13. 如果用户说\"发邮件说XXX\"但没有主题，正文提取为 body，subject 留空放入 missing_fields，不要编造主题。\n\n"
        "输出 JSON schema：\n"
        "{\n"
        '  "route": "chat" | "tool" | "research" | "rag" | "artifact" | "memory" | "skill" | "mixed" | "unknown_tool",\n'
        '  "confidence": number (0.0-1.0),\n'
        '  "tool_calls": [{"name": string, "arguments": object}],\n'
        '  "missing_fields": [{"tool_name": string, "field": string, "question": string}],\n'
        '  "requested_action": string | null,\n'
        '  "reason": string\n'
        "}\n\n"
        "示例 1 — 发邮件完整参数：\n"
        "用户：帮我给 yu@qq.com 发送个邮件，主题是高考，正文是哈哈哈\n"
        "available_tools 包含 email.send\n"
        "输出：{\"route\":\"tool\",\"confidence\":0.98,\"tool_calls\":[{\"name\":\"email.send\","
        "\"arguments\":{\"to\":\"yu@qq.com\",\"subject\":\"高考\",\"body\":\"哈哈哈\"}}],"
        "\"missing_fields\":[],\"requested_action\":\"send_email\",\"reason\":\"用户明确要求发送邮件\"}\n\n"
        "示例 2 — 发邮件缺主题：\n"
        "用户：帮我给 yu@qq.com 发邮件\n"
        "输出：{\"route\":\"tool\",\"confidence\":0.9,\"tool_calls\":[{\"name\":\"email.send\","
        "\"arguments\":{\"to\":\"yu@qq.com\"}}],\"missing_fields\":["
        "{\"tool_name\":\"email.send\",\"field\":\"subject\",\"question\":\"邮件主题是什么？\"},"
        "{\"tool_name\":\"email.send\",\"field\":\"body\",\"question\":\"邮件正文是什么？\"}],"
        "\"requested_action\":\"send_email\",\"reason\":\"用户要发送邮件但缺主题和正文\"}\n\n"
        "示例 3 — 写文件：\n"
        "用户：在 workspace 创建 hello.txt，内容是 hello world\n"
        "输出：{\"route\":\"tool\",\"confidence\":0.95,\"tool_calls\":[{\"name\":\"local_file.write\","
        "\"arguments\":{\"path\":\"hello.txt\",\"content\":\"hello world\"}}],"
        "\"missing_fields\":[],\"requested_action\":\"write_file\",\"reason\":\"用户要创建文件\"}\n\n"
        "示例 4 — 研究：\n"
        "用户：帮我研究一下 LangGraph 最新动态\n"
        "输出：{\"route\":\"research\",\"confidence\":0.9,\"tool_calls\":[],\"missing_fields\":[],"
        "\"requested_action\":null,\"reason\":\"用户要求研究一个主题\"}\n\n"
        "示例 5 — 聊天：\n"
        "用户：你好\n"
        "输出：{\"route\":\"chat\",\"confidence\":0.95,\"tool_calls\":[],\"missing_fields\":[],"
        "\"requested_action\":null,\"reason\":\"用户只是打招呼\"}\n\n"
        f"当前可用的工具列表 (available_tools)：\n{tools_json}\n\n"
        f"用户输入：{user_input}\n\n"
        "现在判断并输出 JSON："
    )


def llm_select_tools(
    db: Session,
    *,
    run_id: int,
    thread_id: str,
    user_id: int,
    user_input: str,
    available_tools: list[dict[str, Any]],
) -> "LLMToolSelectionResult":  # noqa: F821
    """Use LLM to select tools and extract arguments from user natural language.

    Returns a structured result. On any error, returns an empty result (fail-open)
    so callers fall back to keyword-based infer_tool().
    """
    from src.web_app.agent.runtime.intent_schema import LLMToolSelectionResult

    resolution = resolve_model_name("intent", complexity="low")
    started = time.perf_counter()
    output_text = ""
    try:
        model = get_chat_model("intent", complexity="low", temperature=0)
        prompt = _build_tool_selection_prompt(
            user_input=user_input,
            available_tools=available_tools,
        )
        message = model.invoke(prompt)
        output_text = _message_content(message)
        payload = _parse_json(output_text)
        result = LLMToolSelectionResult.model_validate(payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        record_llm_call(
            db,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            node_name="tool_agent",
            purpose="tool_selection",
            provider=resolution.provider,
            model=resolution.model,
            tier=resolution.tier,
            latency_ms=latency_ms,
            status="completed",
            estimated_input_chars=len(prompt),
            estimated_output_chars=len(output_text),
            metadata={
                "input_preview": user_input[:200],
                "confidence": result.confidence,
                "tool_name": result.tool_calls[0].name if result.tool_calls else "",
            },
        )
        return result
    except Exception as exc:
        # fail-open: return empty result, caller falls back to keyword infer_tool
        _record_failed_call(db, run_id, thread_id, user_id, resolution, started, user_input, output_text, str(exc))
        return LLMToolSelectionResult(route="chat", confidence=0.0, reason=f"LLM error: {exc}")
