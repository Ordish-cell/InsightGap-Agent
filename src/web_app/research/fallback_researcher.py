from typing import Any

from src.web_app.research.report_builder import build_markdown_report
from src.web_app.research.schemas import ResearchResult


class FallbackResearcher:
    async def run(self, query: str, context: dict[str, Any], evidence: list[dict[str, Any]], depth: str = "standard") -> ResearchResult:
        usable_evidence = evidence[:8]
        summary = self._summary(query, usable_evidence)
        findings = self._findings(query, usable_evidence)
        risks = self._risks(usable_evidence)
        opportunities = self._opportunities(query, usable_evidence)
        actions = self._actions()
        data = {"summary": summary, "findings": findings, "evidence": usable_evidence, "risks": risks, "opportunities": opportunities, "suggested_actions": actions}
        return ResearchResult(summary=summary, findings=findings, evidence=usable_evidence, risks=risks, opportunities=opportunities, suggested_actions=actions, markdown_report=build_markdown_report(query[:80], data), metadata={"adapter": "fallback", "depth": depth, "evidence_insufficient": not bool(usable_evidence)})

    def _summary(self, query: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return f"Evidence is insufficient to fully answer: {query}. The report is limited to available context."
        snippets = " ".join(item.get("snippet", "") for item in evidence[:3])
        return f"Based on {len(evidence)} evidence items, the research question '{query}' points to a traceable information-gap opportunity. Key evidence: {snippets[:500]}"

    def _findings(self, query: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not evidence:
            return [{"title": "Evidence insufficient", "detail": "No reliable evidence was available, so no high-confidence finding is asserted.", "confidence": 0.2, "evidence_refs": []}]
        return [
            {"title": f"Finding from {item.get('source_type', 'source')}", "detail": item.get("snippet", "")[:300], "confidence": min(float(item.get("score", 0.5)), 0.95), "evidence_refs": [item.get("url") or item.get("title", "")]}
            for item in evidence[:5]
        ]

    def _risks(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        risks = []
        if len(evidence) < 2:
            risks.append({"title": "Limited evidence", "detail": "The evidence set is small; validate with additional sources before acting."})
        if any(float(item.get("score", 0)) < 0.5 for item in evidence):
            risks.append({"title": "Low confidence source", "detail": "At least one source has low credibility or weak retrieval score."})
        return risks or [{"title": "Execution uncertainty", "detail": "The signal may not translate into a useful product or research outcome without follow-up validation."}]

    def _opportunities(self, query: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"title": "Generate a focused report", "detail": "Convert the FeedCard signal into a concise research artifact for later comparison."},
            {"title": "Create a reusable skill draft", "detail": "Capture the repeated workflow: load signal, gather evidence, build GSSC context, produce report."},
            {"title": "Follow the strongest source", "detail": f"Start from: {(evidence[0].get('title') if evidence else query)}"},
        ]

    def _actions(self) -> list[dict[str, Any]]:
        return [
            {"title": "Save report", "detail": "Keep the markdown artifact for later review."},
            {"title": "Add memory", "detail": "Record this research run as episodic memory."},
            {"title": "Review evidence", "detail": "Open sources manually before making product or investment decisions."},
        ]
