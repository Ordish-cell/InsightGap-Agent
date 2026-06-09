"""Adapter that calls the real upstream Open Deep Research graph.

Delegates to ``src/open_deep_research/deep_researcher.py`` without
modifying the upstream directory.  Falls back to ``FallbackResearcher``
only when the upstream graph is unavailable or raises an error.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import traceback
from typing import Any
from uuid import uuid4

from src.web_app.core.config import Settings, get_settings
from src.web_app.research.fallback_researcher import FallbackResearcher
from src.web_app.research.report_builder import build_markdown_report
from src.web_app.research.schemas import ResearchResult

logger = logging.getLogger(__name__)


class OpenDeepResearchConfigError(Exception):
    """Raised when the upstream graph cannot run due to missing configuration."""


class OpenDeepResearchAdapter:
    """Thin wrapper around the upstream ``deep_researcher`` LangGraph graph.

    On success the returned ``ResearchResult.metadata`` will include::

        {"source": "open_deep_research", "engine": "open_deep_research",
         "used_fallback": False, "adapter": "OpenDeepResearchAdapter",
         "odr_enabled": True}

    On failure the caller (``ResearchService``) is expected to catch the
    exception and delegate to ``FallbackResearcher``, recording the error
    in the fallback metadata.
    """

    def __init__(self, settings_obj: Settings | None = None) -> None:
        self._settings: Settings = settings_obj or get_settings()
        self._graph: Any = None  # cached compiled graph

    # ── public API ──────────────────────────────────────────────────────

    async def run_research(
        self,
        query: str,
        *,
        user_id: int | None = None,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        depth: str = "standard",
    ) -> ResearchResult:
        """Execute the upstream Open Deep Research graph.

        Raises ``OpenDeepResearchConfigError`` when required configuration
        (e.g. Tavily API key) is missing, so the caller can fall back.
        """
        self._ensure_config()
        return await self._invoke_graph(
            query=query,
            user_id=user_id,
            run_id=run_id,
            context=context or {},
            evidence=evidence or [],
            depth=depth,
        )

    def health(self) -> dict[str, Any]:
        try:
            return {
                "status": "ok" if self._official_importable() else "degraded",
                "adapter": "OpenDeepResearchAdapter",
                "odr_importable": self._official_importable(),
                "odr_enabled": self._settings.enable_open_deep_research,
            }
        except Exception:
            return {"status": "error", "adapter": "OpenDeepResearchAdapter"}

    # ── helpers ─────────────────────────────────────────────────────────

    def _official_importable(self) -> bool:
        return importlib.util.find_spec("open_deep_research.deep_researcher") is not None

    def _ensure_config(self) -> None:
        """Validate that the upstream graph has the minimum required config."""
        search_api = self._settings.odr_search_api
        if search_api == "tavily" and not os.getenv("TAVILY_API_KEY") and not self._settings.tavily_api_key:
            raise OpenDeepResearchConfigError(
                "TAVILY_API_KEY is required when ODR_SEARCH_API=tavily"
            )

    def _setup_env_for_upstream(self) -> None:
        """Map DashScope credentials to the env vars the upstream expects.

        The upstream ``Configuration`` reads ``OPENAI_API_KEY`` /
        ``OPENAI_BASE_URL`` / ``TAVILY_API_KEY`` from the environment.
        We inject them from our own settings so the user only needs to
        configure credentials once.
        """
        dashscope_key = self._settings.dashscope_api_key or self._settings.aliyun_bailian_api_key or self._settings.agent_llm_api_key or ""
        if dashscope_key:
            os.environ["OPENAI_API_KEY"] = dashscope_key

        dashscope_url = self._settings.dashscope_base_url or self._settings.aliyun_bailian_base_url or self._settings.agent_llm_base_url or ""
        if dashscope_url:
            os.environ["OPENAI_BASE_URL"] = dashscope_url

        if self._settings.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self._settings.tavily_api_key

    # Marker used to satisfy DashScope/Qwen JSON mode and later stripped
    # from the final report if it leaks through.
    #
    # DashScope/Qwen models with thinking enabled return
    #   [{"type":"thinking","text":"..."}, {...}]
    # instead of a plain JSON object, which breaks Pydantic
    # structured-output parsing inside the upstream ODR graph.
    # This guard instructs the model to suppress thinking blocks
    # and return only the JSON object.
    _JSON_GUARD_TEXT = (
        "Compatibility instruction for DashScope/Qwen OpenAI-compatible models. "
        "If any internal Open Deep Research step requests response_format=json_object "
        "or structured output, return exactly one valid JSON object. "
        "Do not return a JSON array. "
        "Do not return thinking blocks. "
        "Do not return content like [{\"type\":\"thinking\",\"text\":\"...\"}]. "
        "Do not include reasoning, analysis, chain-of-thought, or hidden thinking. "
        "Do not wrap JSON in markdown fences. "
        "Return only the JSON object required by the Pydantic schema. "
        "CRITICAL schema field names: "
        "ResearchQuestion requires field research_brief (NOT research_question, NOT question). "
        "ClarifyWithUser requires fields need_clarification, question, verification. "
        "Use exactly the field names defined in each schema, do not guess or reword them. "
        "The word json is intentionally included to satisfy JSON mode requirements."
    )

    async def _invoke_graph(
        self,
        query: str,
        user_id: int | None,
        run_id: str | None,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
        depth: str,
    ) -> ResearchResult:
        self._setup_env_for_upstream()

        from langchain_core.messages import HumanMessage, SystemMessage
        from open_deep_research.deep_researcher import deep_researcher

        thread_id = run_id or str(uuid4())
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "search_api": self._settings.odr_search_api,
                "allow_clarification": self._settings.odr_allow_clarification,
                "max_concurrent_research_units": self._settings.odr_max_concurrent_research_units,
                "max_researcher_iterations": self._settings.odr_max_researcher_iterations,
                "max_react_tool_calls": self._settings.odr_max_react_tool_calls,
                "research_model": self._settings.odr_research_model,
                "summarization_model": self._settings.odr_summarization_model,
                "compression_model": self._settings.odr_compression_model,
                "final_report_model": self._settings.odr_final_report_model,
                "research_model_max_tokens": self._settings.odr_research_model_max_tokens,
                "summarization_model_max_tokens": self._settings.odr_summarization_model_max_tokens,
                "compression_model_max_tokens": self._settings.odr_compression_model_max_tokens,
                "final_report_model_max_tokens": self._settings.odr_final_report_model_max_tokens,
                "max_content_length": self._settings.odr_max_content_length,
            }
        }

        # Include a SystemMessage with "json" so DashScope/Qwen
        # OpenAI-compatible endpoints satisfy the requirement that
        # messages contain "json" when response_format=json_object
        # is used internally by LangChain's with_structured_output().
        #
        # Also wrap the user query with schema field-name hints so
        # Qwen returns research_brief (not research_question) for the
        # ResearchQuestion Pydantic model.
        guarded_query = (
            query
            + "\n\n[System note: for any internal structured output step, "
            + "use the exact Pydantic field names. "
            + "ResearchQuestion uses research_brief, not research_question. "
            + "ClarifyWithUser uses need_clarification, question, verification. "
            + "Do not guess field names from descriptions.]"
        )
        input_state: dict[str, Any] = {
            "messages": [
                SystemMessage(content=self._JSON_GUARD_TEXT),
                HumanMessage(content=guarded_query),
            ]
        }

        logger.info(
            "[OpenDeepResearchAdapter] invoking graph run_id=%s thread_id=%s model=%s search_api=%s json_guard=true",
            run_id,
            thread_id,
            self._settings.odr_research_model,
            self._settings.odr_search_api,
        )

        try:
            output: dict[str, Any] = await asyncio.wait_for(
                deep_researcher.ainvoke(input_state, config=config),
                timeout=self._settings.odr_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise OpenDeepResearchConfigError(
                f"Open Deep Research timed out after {self._settings.odr_timeout_seconds}s"
            )
        except Exception as exc:
            logger.warning(
                "[OpenDeepResearchAdapter] failed run_id=%s error_type=%s error_summary=%s",
                run_id,
                type(exc).__name__,
                str(exc)[:300],
            )
            raise

        return self._parse_output(query, output, context, evidence, depth)

    def _parse_output(
        self,
        query: str,
        output: dict[str, Any],
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
        depth: str,
    ) -> ResearchResult:
        """Extract a ``ResearchResult`` from the raw upstream graph output."""
        report = (
            output.get("final_report")
            or output.get("markdown_report")
            or ""
        )

        # If the graph returned messages, try the last AIMessage
        if not report:
            messages = output.get("messages", [])
            for msg in reversed(messages):
                content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
                if content and len(str(content)) > 50:
                    report = str(content)
                    break

        # Strip JSON guard text and schema hints if they leaked into the final report
        report = report.replace(self._JSON_GUARD_TEXT, "")
        for fragment in self._JSON_GUARD_TEXT.split(". "):
            if len(fragment) > 20:
                report = report.replace(fragment, "")
        # Also strip the inline schema hint appended to the query
        guard_snippets = [
            "[System note: for any internal structured output step,",
            "ResearchQuestion uses research_brief, not research_question.",
            "ClarifyWithUser uses need_clarification, question, verification.",
            "Do not guess field names from descriptions.]",
        ]
        for snippet in guard_snippets:
            report = report.replace(snippet, "")

        research_brief = output.get("research_brief", "")
        notes = output.get("notes", [])
        raw_notes = output.get("raw_notes", [])

        # Build a clean summary from the research brief + first portion of report
        summary_text = research_brief or ""
        if not summary_text and report:
            summary_text = report[:800]

        # Extract sources from notes
        sources: list[dict[str, Any]] = []
        for note in notes:
            if isinstance(note, str) and ("http://" in note or "https://" in note):
                sources.append({"title": "Research source", "url": note, "note": note[:200]})

        usable_evidence = evidence or []
        findings: list[dict[str, Any]] = [
            {"title": "Research completed via Open Deep Research", "detail": summary_text[:500], "confidence": 0.85, "evidence_refs": []}
        ]

        report_text = report if report else build_markdown_report(query[:80], {
            "summary": summary_text,
            "findings": findings,
            "evidence": usable_evidence,
            "risks": [],
            "opportunities": [],
            "suggested_actions": [],
        })

        logger.info(
            "[OpenDeepResearchAdapter] completed run report_chars=%d source_count=%d",
            len(report_text),
            len(sources),
        )

        return ResearchResult(
            summary=summary_text[:800],
            findings=findings,
            evidence=usable_evidence,
            risks=[],
            opportunities=[],
            suggested_actions=[],
            markdown_report=report_text,
            sources=sources,
            metadata={
                "source": "open_deep_research",
                "engine": "open_deep_research",
                "used_fallback": False,
                "adapter": "OpenDeepResearchAdapter",
                "odr_enabled": True,
                "odr_error": None,
                "depth": depth,
                "research_brief": research_brief,
                "notes_count": len(notes),
                "raw_keys": list(output.keys()) if isinstance(output, dict) else [],
            },
        )
