"""Research service — orchestrates deep research runs.

Delegates to ``OpenDeepResearchAdapter`` first; falls back to
``FallbackResearcher`` only when the upstream graph is unavailable
or raises an error.  The fallback result is explicitly marked with
``used_fallback=True`` in its metadata.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import UTC, datetime
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
from src.web_app.research.fallback_researcher import FallbackResearcher
from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter, OpenDeepResearchConfigError
from src.web_app.research.schemas import ResearchRequest, ResearchResult
from src.web_app.services.artifact_service import artifact_service
from src.web_app.services.memory_service import memory_service
from src.web_app.services.rag_service import rag_service
from src.web_app.services.skill_service import skill_service

logger = logging.getLogger(__name__)


def _safe_error_message(exc: BaseException) -> str:
    """Return a user-readable error message — never a full traceback."""
    return f"{type(exc).__name__}: {exc}"


class ResearchService:
    """High-level service for creating and managing research runs."""

    # ── public API used by HTTP endpoints ───────────────────────────────

    def create_research_run(
        self,
        db: Session,
        user_id: int,
        request: ResearchRequest,
        *,
        feed_card: Any = None,
    ) -> dict[str, Any]:
        """Create a research_run row, kick off a background task, and
        return the run_id immediately."""
        research_repo = ResearchRunRepository(db)
        agent_repo = AgentRunRepository(db)

        query = request.query or ""
        if feed_card and not query:
            query = (
                f"Deep research this information gap: {feed_card.title}. "
                f"{feed_card.information_gap or feed_card.one_sentence_value or ''}"
            )

        if not query.strip():
            raise ValueError("Research query is required")

        agent_run = agent_repo.create(
            user_id=user_id,
            run_type="deep_research",
            mode="plan_and_solve",
            status="running",
            user_input=query,
            graph_state={},
        )

        run_id = str(uuid4())
        card_snapshot = request.card_snapshot or {}
        if feed_card and not card_snapshot:
            card_snapshot = {
                "id": getattr(feed_card, "id", None),
                "title": getattr(feed_card, "title", ""),
                "one_sentence_value": getattr(feed_card, "one_sentence_value", ""),
                "information_gap": getattr(feed_card, "information_gap", ""),
                "source_type": getattr(feed_card, "source_type", ""),
                "source_url": getattr(feed_card, "source_url", ""),
                "final_score": getattr(feed_card, "final_score", None),
            }

        research_run = research_repo.create(
            id=run_id,
            user_id=user_id,
            feed_card_id=request.feed_card_id or getattr(feed_card, "id", None),
            agent_run_id=agent_run.id,
            query=query,
            status="running",
            findings=[],
            evidence=[],
            risks=[],
            opportunities=[],
            suggested_actions=[],
            markdown_report="",
            summary="",
            error="",
            metadata_json={
                "source": request.source,
                "engine": "pending",
                "used_fallback": None,
                "odr_enabled": True,
                "odr_error": None,
                "depth": request.depth,
                "feed_card_id": request.feed_card_id or getattr(feed_card, "id", None),
                "card_snapshot": card_snapshot,
            },
        )

        # Fire-and-forget background task
        asyncio.create_task(
            self._run_background(
                run_id=run_id,
                user_id=user_id,
                query=query,
                request=request,
                feed_card=feed_card,
            )
        )

        return self._to_read(research_repo.get_by_user(user_id, run_id))

    def get_research_run(self, db: Session, user_id: int, research_run_id: str) -> dict[str, Any]:
        run = ResearchRunRepository(db).get_by_user(user_id, research_run_id)
        if not run:
            raise ValueError("Research run not found")
        return self._to_read(run)

    def list_research_runs(self, db: Session, user_id: int, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        return [self._to_read(run) for run in ResearchRunRepository(db).list_by_user(user_id, limit, offset)]

    # ── legacy sync helpers (kept for Agent Runtime compat) ────────────

    async def research_feed_card(self, db: Session, user_id: int, card_id: int, request: ResearchRequest) -> dict[str, Any]:
        """Synchronous (blocking) research from a feed card.
        Used by the legacy ``POST /feed/cards/:id/research`` endpoint
        and by the Agent Runtime ``research_agent`` node."""
        card = FeedRepository(db).get_by_user(user_id, card_id)
        if not card:
            raise ValueError("Feed card not found")
        query = request.query or f"Deep research this information gap: {card.title}. {card.information_gap or card.one_sentence_value or ''}"
        return await self._run_sync(db, user_id, request, query=query, feed_card=card)

    async def research_query(self, db: Session, user_id: int, request: ResearchRequest) -> dict[str, Any]:
        """Synchronous (blocking) research from a manual query."""
        if not request.query:
            raise ValueError("Research query is required")
        return await self._run_sync(db, user_id, request, query=request.query, feed_card=None)

    # ── background execution ────────────────────────────────────────────

    async def _run_background(
        self,
        run_id: str,
        user_id: int,
        query: str,
        request: ResearchRequest,
        feed_card: Any = None,
    ) -> None:
        """Execute research in the background and persist the result."""
        # We need a fresh DB session for the background task.
        from src.web_app.db.session import SessionLocal

        db = SessionLocal()
        try:
            research_repo = ResearchRunRepository(db)
            agent_repo = AgentRunRepository(db)

            run = research_repo.get_by_user(user_id, run_id)
            if not run:
                logger.error("[ResearchService] run not found in background task run_id=%s", run_id)
                return

            agent_run = agent_repo.get_by_id(run.agent_run_id) if run.agent_run_id else None

            # ── Collect evidence ────────────────────────────────────
            evidence = self._collect_evidence(user_id, query, feed_card)

            # ── Try Open Deep Research first ────────────────────────
            odr_error: str | None = None
            result: ResearchResult | None = None

            force_fallback = request.force_engine == "fallback"
            use_odr = (not force_fallback)

            if use_odr:
                try:
                    logger.info(
                        "[ResearchService] ODR enabled=True  Trying OpenDeepResearchAdapter run_id=%s query_len=%d",
                        run_id, len(query),
                    )
                    adapter = OpenDeepResearchAdapter()
                    result = await adapter.run_research(
                        query=query,
                        user_id=user_id,
                        run_id=run_id,
                        context={"gssc_context": "", "evidence": evidence},
                        evidence=evidence,
                        depth=request.depth,
                    )
                    result.metadata.update({
                        "source": "open_deep_research",
                        "engine": "open_deep_research",
                        "used_fallback": False,
                        "odr_enabled": True,
                        "adapter": "OpenDeepResearchAdapter",
                        "odr_error": None,
                    })
                    logger.info(
                        "[ResearchService] ODR success used_fallback=False run_id=%s report_chars=%d",
                        run_id, len(result.markdown_report),
                    )

                except OpenDeepResearchConfigError as exc:
                    odr_error = _safe_error_message(exc)
                    logger.warning(
                        "[ResearchService] ODR config error, falling back run_id=%s error=%s",
                        run_id, odr_error,
                    )
                except Exception as exc:
                    odr_error = _safe_error_message(exc)
                    logger.warning(
                        "[ResearchService] ODR failed, fallback enabled run_id=%s error_type=%s error=%s",
                        run_id, type(exc).__name__, odr_error,
                    )

            # ── Fallback if ODR didn't produce a result ─────────────
            if result is None:
                try:
                    logger.info("[ResearchService] Falling back to FallbackResearcher run_id=%s", run_id)
                    fallback = FallbackResearcher()
                    fallback_result = await fallback.run(query, {}, evidence, request.depth)
                    result = fallback_result
                    result.metadata.update({
                        "source": "fallback_researcher",
                        "engine": "fallback_researcher",
                        "used_fallback": True,
                        "odr_enabled": True,
                        "adapter": "FallbackResearcher",
                        "odr_error": odr_error,
                        "odr_error_type": type(OpenDeepResearchConfigError()).__name__ if "ConfigError" in (odr_error or "") else None,
                    })
                    logger.info(
                        "[ResearchService] Fallback success used_fallback=True run_id=%s",
                        run_id,
                    )
                except Exception as fallback_exc:
                    error_msg = _safe_error_message(fallback_exc)
                    logger.error("[ResearchService] Fallback also failed run_id=%s error=%s", run_id, error_msg)
                    research_repo.update(run, status="failed", error=error_msg)
                    if agent_run:
                        agent_repo.update(agent_run, status="failed", error_message=error_msg)
                    return

            # ── Save results ────────────────────────────────────────
            artifact_id = self._save_artifact(db, user_id, run_id, query, result.markdown_report, request) if request.save_artifact else None
            skill_id = self._create_skill(db, user_id, run_id, query, request) if request.create_skill_draft else None

            if request.write_memory:
                try:
                    memory_service.add_memory(
                        user_id=user_id,
                        content=f"深度研究完成：{query}\n摘要：{result.summary}",
                        memory_type="episodic",
                        importance=0.75,
                        metadata={
                            "source_type": "research_run",
                            "source_id": run_id,
                            "feed_card_id": getattr(feed_card, "id", None),
                            "artifact_id": artifact_id,
                        },
                        db=db,
                    )
                except Exception:
                    pass

            completed = datetime.now(UTC).replace(tzinfo=None)
            research_repo.update(
                run,
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
            if agent_run:
                agent_repo.update(agent_run, status="completed", result_summary=result.summary, graph_state={"research_run_id": run_id})

        except Exception as exc:
            logger.error("[ResearchService] unhandled background error run_id=%s error=%s", run_id, str(exc))
        finally:
            db.close()

    # ── synchronous execution (legacy / Agent Runtime path) ─────────────

    async def _run_sync(
        self,
        db: Session,
        user_id: int,
        request: ResearchRequest,
        query: str,
        feed_card: Any = None,
    ) -> dict[str, Any]:
        """Synchronous research execution — blocks until complete.
        Used by the Agent Runtime and the legacy sync API endpoints."""
        research_repo = ResearchRunRepository(db)
        agent_repo = AgentRunRepository(db)

        agent_run = agent_repo.create(
            user_id=user_id,
            run_type="deep_research",
            mode="plan_and_solve",
            status="running",
            user_input=query,
            graph_state={},
        )

        run_id = str(uuid4())
        card_snapshot = request.card_snapshot or {}
        if feed_card and not card_snapshot:
            card_snapshot = {
                "id": getattr(feed_card, "id", None),
                "title": getattr(feed_card, "title", ""),
                "one_sentence_value": getattr(feed_card, "one_sentence_value", ""),
                "information_gap": getattr(feed_card, "information_gap", ""),
                "source_type": getattr(feed_card, "source_type", ""),
                "source_url": getattr(feed_card, "source_url", ""),
                "final_score": getattr(feed_card, "final_score", None),
            }

        research_run = research_repo.create(
            id=run_id,
            user_id=user_id,
            feed_card_id=request.feed_card_id or getattr(feed_card, "id", None),
            agent_run_id=agent_run.id,
            query=query,
            status="running",
            findings=[],
            evidence=[],
            risks=[],
            opportunities=[],
            suggested_actions=[],
            markdown_report="",
            summary="",
            error="",
            metadata_json={
                "source": request.source,
                "engine": "pending",
                "used_fallback": None,
                "odr_enabled": True,
                "depth": request.depth,
                "feed_card_id": request.feed_card_id or getattr(feed_card, "id", None),
                "card_snapshot": card_snapshot,
            },
        )

        try:
            evidence = self._collect_evidence(user_id, query, feed_card)
            result = await self._execute_research(user_id, run_id, query, request, evidence)

            artifact_id = self._save_artifact(db, user_id, run_id, query, result.markdown_report, request) if request.save_artifact else None
            skill_id = self._create_skill(db, user_id, run_id, query, request) if request.create_skill_draft else None

            if request.write_memory:
                try:
                    memory_service.add_memory(
                        user_id=user_id,
                        content=f"深度研究完成：{query}\n摘要：{result.summary}",
                        memory_type="episodic",
                        importance=0.75,
                        metadata={
                            "source_type": "research_run",
                            "source_id": run_id,
                            "feed_card_id": getattr(feed_card, "id", None),
                            "artifact_id": artifact_id,
                        },
                        db=db,
                    )
                except Exception:
                    pass

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
            agent_repo.update(agent_run, status="completed", result_summary=result.summary, graph_state={"research_run_id": run_id})

        except Exception as exc:
            error_msg = _safe_error_message(exc)
            research_repo.update(research_run, status="failed", error=error_msg)
            agent_repo.update(agent_run, status="failed", error_message=error_msg)

        return self._to_read(research_repo.get_by_user(user_id, run_id))

    async def _execute_research(
        self,
        user_id: int,
        run_id: str,
        query: str,
        request: ResearchRequest,
        evidence: list[dict[str, Any]],
    ) -> ResearchResult:
        """Try ODR first, fall back on failure."""
        odr_error: str | None = None

        force_fallback = request.force_engine == "fallback"
        if not force_fallback:
            try:
                logger.info("[ResearchService] ODR enabled=True  Trying OpenDeepResearchAdapter run_id=%s", run_id)
                adapter = OpenDeepResearchAdapter()
                result = await adapter.run_research(
                    query=query,
                    user_id=user_id,
                    run_id=run_id,
                    context={"evidence": evidence},
                    evidence=evidence,
                    depth=request.depth,
                )
                result.metadata.update({
                    "source": "open_deep_research",
                    "engine": "open_deep_research",
                    "used_fallback": False,
                    "odr_enabled": True,
                    "adapter": "OpenDeepResearchAdapter",
                    "odr_error": None,
                })
                logger.info("[ResearchService] ODR success used_fallback=False run_id=%s", run_id)
                return result
            except OpenDeepResearchConfigError as exc:
                odr_error = _safe_error_message(exc)
                logger.warning("[ResearchService] ODR config error run_id=%s error=%s", run_id, odr_error)
            except Exception as exc:
                odr_error = _safe_error_message(exc)
                logger.warning("[ResearchService] ODR failed run_id=%s error=%s", run_id, odr_error)

        fallback = FallbackResearcher()
        fallback_result = await fallback.run(query, {}, evidence, request.depth)
        fallback_result.metadata.update({
            "source": "fallback_researcher",
            "engine": "fallback_researcher",
            "used_fallback": True,
            "odr_enabled": True,
            "adapter": "FallbackResearcher",
            "odr_error": odr_error,
        })
        logger.info("[ResearchService] Fallback success used_fallback=True run_id=%s", run_id)
        return fallback_result

    # ── helpers ─────────────────────────────────────────────────────────

    def _collect_evidence(self, user_id: int, query: str, feed_card: Any = None) -> list[dict[str, Any]]:
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
        artifact = ArtifactRepository(db).create(
            user_id=user_id,
            run_id=None,
            artifact_type="research_report",
            title=f"Research Report: {query[:80]}",
            file_path=file_path,
            metadata_json={
                "source_type": "deep_research",
                "source_id": research_run_id,
                "query": query,
                "depth": request.depth,
            },
        )
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

    def _to_read(self, run: Any) -> dict[str, Any]:
        if run is None:
            return {}
        return {
            "id": run.id,
            "user_id": run.user_id,
            "feed_card_id": run.feed_card_id,
            "agent_run_id": run.agent_run_id,
            "query": run.query,
            "status": run.status,
            "summary": run.summary or "",
            "findings": run.findings or [],
            "evidence": run.evidence or [],
            "risks": run.risks or [],
            "opportunities": run.opportunities or [],
            "suggested_actions": run.suggested_actions or [],
            "markdown_report": run.markdown_report or "",
            "sources": (run.metadata_json or {}).get("sources", []),
            "artifact_id": run.artifact_id,
            "skill_draft_id": run.skill_draft_id,
            "error": run.error or "",
            "error_message": run.error or "",
            "metadata": run.metadata_json or {},
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }


research_service = ResearchService()


def run_deep_research(query: str, user_id: int = 1) -> dict:
    return {"query": query, "user_id": user_id, "status": "use ResearchService.create_research_run"}
