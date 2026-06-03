from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_step
from src.web_app.agent.runtime.router import route_user_input
from src.web_app.agent.runtime.state import AgentRuntimeState
from src.web_app.mcp.tool_router import infer_tool
from src.web_app.context.builder import ContextBuilder
from src.web_app.core.constants import L3_EXTERNAL_WRITE, L4_HIGH_RISK
from src.web_app.db.repositories.approval_repository import ApprovalRepository
from src.web_app.db.repositories.artifact_repository import ArtifactRepository
from src.web_app.db.repositories.profile_repository import ProfileRepository
from src.web_app.research.schemas import ResearchRequest
from src.web_app.services.artifact_service import artifact_service
from src.web_app.services.memory_service import memory_service
from src.web_app.services.permission_service import PermissionGuard
from src.web_app.services.rag_service import rag_service
from src.web_app.services.research_service import research_service
from src.web_app.services.skill_service import skill_service
from src.web_app.services.mcp_service import mcp_service

EXTERNAL_WRITE_TERMS = ("发邮件", "发送邮件", "邮件", "评论", "发布", "提交表单", "email", "send", "post", "submit")
HIGH_RISK_TERMS = ("删除", "支付", "付款", "转账", "delete", "payment")


class RuntimeNodes:
    def __init__(self, db: Session, payload: dict[str, Any]):
        self.db = db
        self.payload = payload

    async def permission_guard(self, state: AgentRuntimeState) -> AgentRuntimeState:
        text = state.get("user_input", "")
        permission_level = "L0_READ_ONLY"
        if any(term in text for term in HIGH_RISK_TERMS):
            permission_level = L4_HIGH_RISK
        elif any(term in text for term in EXTERNAL_WRITE_TERMS):
            permission_level = L3_EXTERNAL_WRITE

        decision = PermissionGuard().check_tool_call("agent_runtime_task", permission_level)
        state["permission"] = {"level": permission_level, **decision}
        if decision["requires_approval"]:
            ApprovalRepository(self.db).create(
                user_id=state["user_id"],
                run_id=state["run_id"],
                approval_type="agent_runtime",
                title="Agent runtime approval required",
                description=text,
                payload={"user_input": text, "permission_level": permission_level},
            )
            state["route"] = "approval"
            state["status"] = "waiting_approval"
            state["final_output"] = "This task requires approval before execution."
        elif not decision["allowed"]:
            state["route"] = "blocked"
            state["status"] = "failed"
            state["error"] = decision["reason"]
            state["final_output"] = "High risk operation denied."
        record_step(self.db, state["run_id"], "permission_guard", "permission", {"user_input": text}, {"permission": state["permission"], "route": state.get("route")})
        return state

    async def router(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        route = route_user_input(state.get("user_input", ""), self.payload)
        state["route"] = route
        record_step(self.db, state["run_id"], "router", "route", {"user_input": state.get("user_input", "")}, {"route": route})
        return state

    async def context_builder(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"}:
            return state
        profile = ProfileRepository(self.db).get_or_create_default(state["user_id"])
        memories = memory_service.search_memory(state["user_id"], state.get("user_input", ""), min_importance=0.2, db=self.db)[:5]
        context_text = ContextBuilder().build(
            {
                "task": state.get("user_input", ""),
                "route": state.get("route"),
                "profile": {"segment": profile.segment, "goals": profile.goals, "interests": profile.explicit_interests},
                "memory": memories,
                "output_contract": "Return structured status, final_output, artifacts, memory_updates, skill_drafts, and evidence when available.",
            }
        )
        state["context"] = {"gssc_context": context_text, "memory_count": len(memories)}
        record_step(self.db, state["run_id"], "context_builder", "context", {"route": state.get("route")}, {"memory_count": len(memories), "context": context_text})
        return state

    async def research(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "research":
            return state
        request = ResearchRequest(
            query=self.payload.get("query") or state.get("user_input", ""),
            depth=self.payload.get("depth", "standard"),
            save_artifact=self.payload.get("save_artifact", True),
            write_memory=self.payload.get("write_memory", True),
            create_skill_draft=self.payload.get("create_skill_draft", True),
        )
        feed_card_id = self.payload.get("feed_card_id")
        if feed_card_id:
            result = await research_service.research_feed_card(self.db, state["user_id"], int(feed_card_id), request)
        else:
            result = await research_service.research_query(self.db, state["user_id"], request)
        state["research"] = result
        state["final_output"] = result.get("summary") or result.get("status", "")
        state.setdefault("artifacts", [])
        if result.get("artifact_id"):
            state["artifacts"].append({"id": result["artifact_id"], "type": "research_report"})
        if result.get("skill_draft_id"):
            state.setdefault("skill_drafts", []).append({"id": result["skill_draft_id"], "source": "research"})
        record_step(self.db, state["run_id"], "research", "deep_research", {"feed_card_id": feed_card_id, "query": request.query}, {"research_run_id": result.get("id"), "status": result.get("status")})
        return state

    async def rag(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "rag":
            return state
        result = rag_service.ask(state["user_id"], state.get("user_input", ""), top_k=int(self.payload.get("top_k", 5)))
        state["rag"] = result
        state["final_output"] = result.get("answer", "")
        record_step(self.db, state["run_id"], "rag", "rag_ask", {"query": state.get("user_input", "")}, {"answer_mode": result.get("answer_mode"), "evidence_count": len(result.get("evidence", []))})
        return state

    async def artifact(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "artifact":
            return state
        content = state.get("final_output") or state.get("user_input", "")
        filename = f"agent_run_{state['run_id']}.md"
        file_path = artifact_service.save_text_artifact(state["user_id"], filename, content)
        item = ArtifactRepository(self.db).create(user_id=state["user_id"], run_id=state["run_id"], artifact_type="agent_output", title=f"Agent Output {state['run_id']}", file_path=file_path, metadata_json={"route": state.get("route")})
        artifact = {"id": item.id, "type": item.artifact_type, "file_path": item.file_path}
        state.setdefault("artifacts", []).append(artifact)
        state["final_output"] = f"Artifact saved: {item.title}"
        record_step(self.db, state["run_id"], "artifact", "save_artifact", {"filename": filename}, {"artifact": artifact})
        return state

    async def skill_librarian(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "skill":
            return state
        draft = skill_service.create_skill_draft_from_run(
            state["run_id"],
            user_id=state["user_id"],
            db=self.db,
            payload={
                "name": f"Agent runtime skill {state['run_id']}",
                "description": "Draft generated from a user-requested reusable workflow.",
                "trigger_text": state.get("user_input", ""),
                "tool_plan": ["permission_guard", "router", "context_builder", "runtime_node"],
                "eval_checks": ["deterministic_fallback", "no_external_write"],
            },
        )
        state.setdefault("skill_drafts", []).append(draft)
        state["final_output"] = f"Skill draft created: {draft['name']}"
        record_step(self.db, state["run_id"], "skill_librarian", "create_skill_draft", {"run_id": state["run_id"]}, {"skill_id": draft.get("id")})
        return state

    async def tool(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") != "tool":
            return state
        tool_name, tool_input = infer_tool(state.get("user_input", ""), self.payload)
        if not tool_name:
            state["status"] = "failed"
            state["error"] = "tool_not_found"
            state["final_output"] = "No MCP tool was selected."
            record_step(self.db, state["run_id"], "tool", "mcp_call", {"tool_name": None}, {"status": "failed", "error": state["error"]}, status="failed")
            return state
        result = mcp_service.call_tool(self.db, state["user_id"], tool_name, tool_input, agent_run_id=state["run_id"], dry_run=bool(self.payload.get("dry_run", False)))
        state["tool_call"] = result
        if result["status"] == "waiting_approval":
            state["status"] = "waiting_approval"
            state["final_output"] = f"MCP tool {tool_name} is waiting for approval."
        elif result["status"] in {"failed", "blocked"}:
            state["status"] = "failed"
            state["error"] = result.get("error", "")
            state["final_output"] = f"MCP tool {tool_name} failed: {state['error']}"
        else:
            state["final_output"] = f"MCP tool {tool_name} completed."
        record_step(self.db, state["run_id"], "tool", "mcp_call", {"tool_name": tool_name, "input": tool_input}, {"tool_call_id": result.get("id"), "status": result.get("status")}, status="completed" if result["status"] == "completed" else result["status"])
        return state

    async def memory_writer(self, state: AgentRuntimeState) -> AgentRuntimeState:
        if state.get("route") in {"approval", "blocked"} or state.get("status") == "waiting_approval":
            return state
        content = state.get("final_output") or state.get("user_input", "")
        memory = memory_service.add_memory(
            user_id=state["user_id"],
            content=f"Agent runtime completed route={state.get('route')}: {content}",
            memory_type="episodic",
            importance=0.6,
            metadata={"source_type": "agent_run", "source_id": state["run_id"], "route": state.get("route")},
            db=self.db,
        )
        state.setdefault("memory_updates", []).append(memory)
        record_step(self.db, state["run_id"], "memory_writer", "write_memory", {"route": state.get("route")}, {"memory_id": memory.get("id")})
        return state

    async def evaluator(self, state: AgentRuntimeState) -> AgentRuntimeState:
        status = state.get("status") or ("failed" if state.get("error") else "completed")
        if state.get("route") == "approval":
            status = "waiting_approval"
        state["status"] = status
        state["evaluation"] = {
            "route": state.get("route"),
            "status": status,
            "has_output": bool(state.get("final_output")),
            "artifact_count": len(state.get("artifacts", [])),
            "memory_count": len(state.get("memory_updates", [])),
            "skill_count": len(state.get("skill_drafts", [])),
            "tool_call_id": (state.get("tool_call") or {}).get("id"),
        }
        if not state.get("final_output") and status == "completed":
            state["final_output"] = "Agent runtime completed."
        record_step(self.db, state["run_id"], "evaluator", "evaluate", {"route": state.get("route")}, {"evaluation": state["evaluation"]})
        return state
