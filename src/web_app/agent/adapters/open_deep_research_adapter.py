"""Agent Runtime adapter that delegates to the real ResearchService.

Used by the ``research_agent`` node inside the LangGraph Agent Runtime.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class OpenDeepResearchAdapter:
    """Thin adapter used by the Agent Runtime to run deep research.

    Delegates to ``ResearchService.research_query`` (the synchronous path)
    so the same ODR-first-with-fallback logic is shared.

    The adapter is called from synchronous code inside the Agent Runtime,
    but ``research_query`` is async.  We run it in a new event loop on a
    background thread to avoid conflicts with the LangGraph event loop.
    """

    def run_research(self, query: str, context: Any, user_id: int, config: dict[str, Any]) -> dict[str, Any]:
        """Run deep research synchronously for the Agent Runtime."""

        def _run_in_thread() -> dict[str, Any]:
            try:
                from src.web_app.db.session import SessionLocal
                from src.web_app.research.schemas import ResearchRequest
                from src.web_app.services.research_service import research_service

                db = SessionLocal()
                try:
                    request = ResearchRequest(
                        query=query,
                        depth=config.get("depth", "standard"),
                        source="manual",
                        save_artifact=config.get("save_artifact", True),
                        write_memory=config.get("write_memory", True),
                        create_skill_draft=config.get("create_skill_draft", True),
                    )
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(
                            research_service.research_query(db, user_id, request)
                        )
                    finally:
                        loop.close()
                    return {
                        "summary": result.get("summary", ""),
                        "findings": result.get("findings", []),
                        "evidence": result.get("evidence", []),
                        "risks": result.get("risks", []),
                        "opportunities": result.get("opportunities", []),
                        "suggested_actions": result.get("suggested_actions", []),
                        "markdown_report": result.get("markdown_report", ""),
                        "sources": result.get("sources", []),
                        "user_id": user_id,
                        "mode": "open_deep_research",
                        "metadata": result.get("metadata", {}),
                        "artifact_id": result.get("artifact_id"),
                        "skill_draft_id": result.get("skill_draft_id"),
                    }
                finally:
                    db.close()
            except Exception:
                logger.exception("[AgentRuntime ODR adapter] research failed, returning fallback")
                return {
                    "summary": f"Research could not be completed for: {query}",
                    "findings": [],
                    "evidence": [],
                    "risks": [],
                    "opportunities": [],
                    "suggested_actions": [],
                    "user_id": user_id,
                    "mode": "fallback",
                    "metadata": {"source": "fallback", "engine": "fallback", "used_fallback": True},
                }

        result_container: dict[str, Any] = {}
        error_container: Exception | None = None

        def _target() -> None:
            nonlocal error_container
            try:
                result_container.update(_run_in_thread())
            except Exception as exc:
                error_container = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=300)  # 5 min timeout

        if error_container:
            raise error_container
        if not result_container:
            return {
                "summary": f"Research timed out for: {query}",
                "findings": [],
                "evidence": [],
                "risks": [],
                "opportunities": [],
                "suggested_actions": [],
                "user_id": user_id,
                "mode": "timeout",
                "metadata": {"source": "timeout", "engine": "timeout", "used_fallback": True},
            }
        return result_container
