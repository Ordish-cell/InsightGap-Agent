from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from src.web_app.context.builder import ContextBuilder
from src.web_app.db.repositories.agent_repository import AgentRunRepository, AgentStepRepository
from src.web_app.db.repositories.artifact_repository import ArtifactRepository
from src.web_app.db.repositories.feed_repository import FeedRepository
from src.web_app.db.repositories.profile_repository import ProfileRepository
from src.web_app.db.repositories.research_repository import ResearchRunRepository
from src.web_app.research.evidence_builder import evidence_from_feed_card, evidence_from_rag_results
from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter
from src.web_app.research.schemas import ResearchRequest, ResearchRunRead
from src.web_app.services.artifact_service import artifact_service
from src.web_app.services.memory_service import memory_service
from src.web_app.services.rag_service import rag_service
from src.web_app.services.skill_service import skill_service


class ResearchService:
    async def research_feed_card(self, db: Session, user_id: int, card_id: int, request: ResearchRequest) -> dict[str, Any]:
        card = FeedRepository(db).get_by_user(user_id, card_id)
        if not card:
            raise ValueError("Feed card not found")
        query = request.query or f"Deep research this information gap: {card.title}. {card.information_gap}"
        return await self._run(db, user_id, request, query=query, feed_card=card)

    async def research_query(self, db: Session, user_id: int, request: ResearchRequest) -> dict[str, Any]:
        if not request.query:
            raise ValueError("Research query is required")
        return await self._run(db, user_id, request, query=request.query, feed_card=None)

    def get_research_run(self, db: Session, user_id: int, research_run_id: str) -> dict[str, Any]:
        run = ResearchRunRepository(db).get_by_user(user_id, research_run_id)
        if not run:
            raise ValueError("Research run not found")
        return self._to_read(run)

    def list_research_runs(self, db: Session, user_id: int, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        return [self._to_read(run) for run in ResearchRunRepository(db).list_by_user(user_id, limit, offset)]

    async def _run(self, db: Session, user_id: int, request: ResearchRequest, query: str, feed_card=None) -> dict[str, Any]:
        research_repo = ResearchRunRepository(db)
        agent_repo = AgentRunRepository(db)
        agent_run = agent_repo.create(user_id=user_id, run_type="deep_research", mode="plan_and_solve", status="running", user_input=query, graph_state={})
        research_run = research_repo.create(id=str(uuid4()), user_id=user_id, feed_card_id=getattr(feed_card, "id", None), agent_run_id=agent_run.id, query=query, status="pending", findings=[], evidence=[], risks=[], opportunities=[], suggested_actions=[], markdown_report="", summary="", error="", metadata_json={"depth": request.depth})
        try:
            research_repo.update(research_run, status="running")
            evidence = self._collect_evidence(user_id, query, feed_card)
            profile = ProfileRepository(db).get_or_create_default(user_id)
            context_text = ContextBuilder().build(
                {
                    "profile": {"segment": profile.segment, "goals": profile.goals, "interests": profile.explicit_interests},
                    "task": query,
                    "evidence": evidence,
                    "information_gap": getattr(feed_card, "information_gap", ""),
                    "output_contract": '{"summary": "...", "findings": [], "evidence": [], "risks": [], "opportunities": [], "suggested_actions": [], "markdown_report": "..."}',
                }
            )
            AgentStepRepository(db).create(run_id=agent_run.id, node_name="build_gssc_context", agent_name="research_service", action_type="context", input={"query": query}, output={"context": context_text}, status="completed", started_at=datetime.now(UTC), ended_at=datetime.now(UTC))
            result = await OpenDeepResearchAdapter().run_research(query=query, context={"gssc_context": context_text, "evidence": evidence}, evidence=evidence, depth=request.depth)
            artifact_id = self._save_artifact(db, user_id, research_run.id, query, result.markdown_report, request) if request.save_artifact else None
            skill_id = self._create_skill(db, user_id, research_run.id, query, request) if request.create_skill_draft else None
            if request.write_memory:
                memory_service.add_memory(user_id=user_id, content=f"用户围绕 Deep Research 完成研究：{query}\n摘要：{result.summary}", memory_type="episodic", importance=0.75, metadata={"source_type": "research_run", "source_id": research_run.id, "feed_card_id": getattr(feed_card, "id", None), "artifact_id": artifact_id}, db=db)
            completed = datetime.now(UTC).replace(tzinfo=None)
            research_repo.update(
                research_run,
                status="completed",
                summary=result.summary,
                findings=result.findings,
                evidence=result.evidence,
                risks=result.risks,
                opportunities=result.opportunities,
                suggested_actions=result.suggested_actions,
                markdown_report=result.markdown_report,
                artifact_id=artifact_id,
                skill_draft_id=skill_id,
                metadata_json=result.metadata,
                completed_at=completed,
            )
            agent_repo.update(agent_run, status="completed", result_summary=result.summary, graph_state={"research_run_id": research_run.id})
            return self._to_read(research_run)
        except Exception as exc:
            research_repo.update(research_run, status="failed", error=str(exc))
            agent_repo.update(agent_run, status="failed", error_message=str(exc))
            return self._to_read(research_run)

    def _collect_evidence(self, user_id: int, query: str, feed_card=None) -> list[dict[str, Any]]:
        evidence = evidence_from_feed_card(feed_card) if feed_card else []
        try:
            rag_results = rag_service.search(user_id=user_id, query=query, top_k=5, min_score=0.2).get("results", [])
            evidence.extend(evidence_from_rag_results(rag_results))
        except Exception:
            pass
        return evidence

    def _save_artifact(self, db: Session, user_id: int, research_run_id: str, query: str, markdown: str, request: ResearchRequest) -> int:
        filename = f"research_{research_run_id}.md"
        file_path = artifact_service.save_text_artifact(user_id, filename, markdown)
        artifact = ArtifactRepository(db).create(user_id=user_id, run_id=None, artifact_type="research_report", title=f"Research Report: {query[:80]}", file_path=file_path, metadata_json={"source_type": "deep_research", "source_id": research_run_id, "query": query, "depth": request.depth})
        return artifact.id

    def _create_skill(self, db: Session, user_id: int, research_run_id: str, query: str, request: ResearchRequest) -> int:
        draft = skill_service.create_skill_draft_from_run(
            run_id=0,
            user_id=user_id,
            db=db,
            payload={
                "name": "基于信息差生成 Deep Research 报告",
                "description": "当用户点击 FeedCard 并希望进一步验证机会时，执行结构化研究并生成报告。",
                "trigger_text": "用户要求对某条信息差进行深度研究、调研、验证机会、生成报告",
                "input_schema": {"feed_card_id": "string", "query": "string optional", "depth": "quick|standard|deep"},
                "context_recipe": ["FeedCard", "UserProfile", "RAG evidence", "Relevant Memory", "Information Gap Signals"],
                "tool_plan": ["load_feed_card", "rag_search", "build_gssc_context", "open_deep_research", "save_artifact", "write_memory"],
                "output_schema": {"summary": "string", "findings": "list", "evidence": "list", "risks": "list", "opportunities": "list", "suggested_actions": "list", "artifact_id": "string"},
                "safety_level": "read_only",
                "eval_checks": ["must_have_evidence", "must_have_markdown_report", "must_not_modify_external_systems"],
            },
        )
        return int(draft["id"])

    def _to_read(self, run) -> dict[str, Any]:
        return {
            "id": run.id,
            "user_id": run.user_id,
            "feed_card_id": run.feed_card_id,
            "agent_run_id": run.agent_run_id,
            "query": run.query,
            "status": run.status,
            "summary": run.summary,
            "findings": run.findings or [],
            "evidence": run.evidence or [],
            "risks": run.risks or [],
            "opportunities": run.opportunities or [],
            "suggested_actions": run.suggested_actions or [],
            "markdown_report": run.markdown_report,
            "artifact_id": run.artifact_id,
            "skill_draft_id": run.skill_draft_id,
            "error": run.error,
            "metadata": run.metadata_json or {},
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }


research_service = ResearchService()


def run_deep_research(query: str, user_id: int = 1) -> dict:
    return {"query": query, "user_id": user_id, "status": "use ResearchService.research_query"}
