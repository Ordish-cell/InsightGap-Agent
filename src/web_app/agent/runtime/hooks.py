"""Explicit post-run save options share tool services and durable receipts."""

from uuid import NAMESPACE_URL, uuid5

from src.web_app.core.config import settings
from .capabilities import observe
from .tools import execute_tool


async def save_outputs(nodes, state, answer):
    if (
        state.get("termination_reason")
        or state.get("writes_denied")
        or state.get("error")
    ):
        return
    policy = state.get("save_policy", {})
    if any(policy.values()):
        from src.web_app.agent.runtime.chat_control import disable_control

        disable_control(nodes.db, state["run_id"])
    attempted = {
        r["capability"]
        for r in state.get("observations", [])
        if not (
            r["capability"] == "skill" and r.get("data", {}).get("operation") == "match"
        )
    }
    saved_tools = {
        "artifact_mcp.create_text_artifact": "artifact",
        "memory_mcp.add": "memory_write",
        "memory_mcp.extract": "memory_write",
        "skill_mcp.create_draft": "skill",
    }
    attempted.update(
        saved_tools[c["tool_name"]]
        for c in state.get("tool_calls", [])
        if c.get("tool_name") in saved_tools
    )
    # An explicit call (even failed) owns its result; a hook must not repeat it.
    for kind, flag, name, output_key in (
        ("artifact", "save_artifact", "artifact_mcp.create_text_artifact", "artifacts"),
        ("memory_write", "write_memory", "memory_mcp.extract", "memory_updates"),
        ("skill", "create_skill_draft", "skill_mcp.create_draft", "skill_drafts"),
    ):
        if state.get("writes_denied"):
            break
        if not policy.get(flag) or kind in attempted or state.get(output_key):
            continue
        if state["runtime_budget"]["tool_calls"] >= settings.agent_max_tool_calls:
            state.setdefault("final_warnings", []).append(
                f"{kind}: 保存未执行，工具预算已用尽。"
            )
            continue
        if kind == "artifact":
            inputs = {
                "title": state["user_input"][:120],
                "content": answer,
                "artifact_type": "markdown_report",
            }
        elif kind == "memory_write":
            inputs = {
                "user_input": state["user_input"],
                "agent_output": answer,
                "page_context": state.get("page_context", {}),
                "thread_id": state.get("thread_id", ""),
            }
        else:
            from src.web_app.services.skill_service import skill_service

            reuse = skill_service.evaluate_reusability(state)
            state["skill_reuse"] = reuse
            if not reuse.get("should_create"):
                state.setdefault("final_warnings", []).append(
                    "Skill：本轮结果未达到现有复用标准，未创建草稿。"
                )
                continue
            inputs = {
                "name": state["user_input"][:120],
                "description": answer[:2000],
                "trigger_text": state["user_input"],
                "tool_plan": [
                    {"tool_name": c.get("tool_name"), "input": c.get("input", {})}
                    for c in state.get("tool_calls", [])
                    if c.get("status") == "completed"
                ],
            }
        original = state.get("current_action")
        state["current_action"] = {
            "action": kind,
            "arguments": inputs,
            "action_id": uuid5(
                NAMESPACE_URL, f"insightgap:{state['run_id']}:save:{kind}"
            ).hex,
        }
        state["runtime_budget"]["tool_calls"] += 1
        try:
            result = await execute_tool(nodes, state, name, inputs)
            observe(nodes, state, result)
            if result.status != "ok":
                state.setdefault("final_warnings", []).append(
                    f"{kind}: {result.status}; {result.error or '未确认保存成功'}"
                )
        except Exception as exc:
            state.setdefault("final_warnings", []).append(
                f"{kind}: 保存失败 ({type(exc).__name__})。"
            )
        finally:
            state["current_action"] = original
