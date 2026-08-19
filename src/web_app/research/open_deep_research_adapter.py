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
        from langchain_core.messages import HumanMessage, SystemMessage
        from open_deep_research.deep_researcher import deep_researcher
        from src.web_app.agent.llm.context import get_model_context
        from src.web_app.agent.llm.factory import normalize_model_endpoint

        model_context = get_model_context()
        provider_prefix = {
            "anthropic_messages": "anthropic",
            "google_generate_content": "google_genai",
            "openai_chat_completions": "openai",
            "openai_responses": "openai",
            "ollama_chat": "openai",
        }.get(model_context.protocol, "openai")
        if model_context.provider == "azure_openai":
            provider_prefix = "azure_openai"
        model_spec = f"{provider_prefix}:{model_context.model}"
        model_api_key = str(model_context.secrets.get("api_key") or "not-required")
        model_base_url = normalize_model_endpoint(str(model_context.config.get("base_url") or ""), model_context.protocol)
        if model_context.protocol == "ollama_chat" and model_base_url:
            model_base_url = model_base_url.rstrip("/") + "/v1"
        headers = dict(model_context.config.get("custom_headers") or {})
        auth_header = str(model_context.config.get("auth_header") or "Authorization")
        if model_context.provider == "custom" and auth_header != "Authorization" and model_api_key != "not-required":
            headers[auth_header] = model_api_key
            model_api_key = "not-required"
        if model_context.provider == "openrouter":
            if model_context.config.get("site_url"):
                headers["HTTP-Referer"] = str(model_context.config["site_url"])
            if model_context.config.get("app_name"):
                headers["X-OpenRouter-Title"] = str(model_context.config["app_name"])
        model_extra: dict[str, Any] = {}
        if model_base_url:
            model_extra["base_url"] = model_base_url
        if model_context.provider == "azure_openai":
            model_extra.update({
                "azure_endpoint": model_context.config.get("endpoint"),
                "azure_deployment": model_context.config.get("deployment") or model_context.model,
                "api_version": model_context.config.get("api_version"),
            })
        if model_context.protocol == "openai_responses":
            model_extra["use_responses_api"] = True
        if headers:
            header_key = "additional_headers" if model_context.protocol == "google_generate_content" else "default_headers"
            model_extra[header_key] = headers

        thread_id = run_id or str(uuid4())
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "search_api": self._settings.odr_search_api,
                "allow_clarification": self._settings.odr_allow_clarification,
                "max_concurrent_research_units": self._settings.odr_max_concurrent_research_units,
                "max_researcher_iterations": self._settings.odr_max_researcher_iterations,
                "max_react_tool_calls": self._settings.odr_max_react_tool_calls,
                "research_model": model_spec,
                "summarization_model": model_spec,
                "compression_model": model_spec,
                "final_report_model": model_spec,
                "model_api_key": model_api_key,
                "model_extra": model_extra,
                "apiKeys": {"TAVILY_API_KEY": self._settings.tavily_api_key},
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
            model_spec,
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
