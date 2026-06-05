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
