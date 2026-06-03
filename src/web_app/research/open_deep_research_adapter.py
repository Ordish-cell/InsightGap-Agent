from typing import Any
import importlib.util

from src.web_app.core.config import Settings, settings
from src.web_app.research.fallback_researcher import FallbackResearcher
from src.web_app.research.schemas import ResearchResult


class OpenDeepResearchAdapter:
    def __init__(self, settings_obj: Settings | None = None):
        self.settings = settings_obj or settings
        self.fallback = FallbackResearcher()

    async def run_research(self, query: str, context: dict[str, Any], evidence: list[dict[str, Any]], depth: str = "standard") -> ResearchResult:
        if self._can_try_official():
            try:
                result = await self._run_official(query, context, depth)
                if result:
                    return result
            except Exception:
                pass
        return await self.fallback.run(query, context, evidence, depth)

    def health(self) -> dict[str, Any]:
        return {"status": "ok" if self._official_importable() else "degraded", "adapter": "available", "fallback_enabled": True}

    def _can_try_official(self) -> bool:
        return self.settings.open_deep_research_mode == "real" and self._official_importable()

    def _official_importable(self) -> bool:
        return importlib.util.find_spec("open_deep_research.deep_researcher") is not None

    async def _run_official(self, query: str, context: dict[str, Any], depth: str) -> ResearchResult | None:
        from langchain_core.messages import HumanMessage
        from open_deep_research.deep_researcher import deep_researcher

        output = await deep_researcher.ainvoke({"messages": [HumanMessage(content=query)]})
        report = output.get("final_report") or output.get("messages", [""])[-1]
        evidence = context.get("evidence", [])
        data = {"summary": str(report)[:800], "findings": [], "evidence": evidence, "risks": [], "opportunities": [], "suggested_actions": []}
        from src.web_app.research.report_builder import build_markdown_report

        return ResearchResult(markdown_report=build_markdown_report(query[:80], {**data, "summary": str(report)}), metadata={"adapter": "official", "depth": depth}, **data)
