from typing import Any


def build_markdown_report(title: str, result: dict[str, Any]) -> str:
    findings = result.get("findings", [])
    evidence = result.get("evidence", [])
    opportunities = result.get("opportunities", [])
    risks = result.get("risks", [])
    actions = result.get("suggested_actions", [])
    return "\n\n".join(
        [
            f"# Research Report: {title}",
            "## 1. Executive Summary\n\n" + (result.get("summary") or "No summary available."),
            "## 2. Why This Matters\n\nThis research expands a FeedCard information gap into a traceable report grounded in available evidence.",
            "## 3. Key Findings\n\n" + _bullets(findings, "title", "detail"),
            "## 4. Evidence\n\n" + _evidence(evidence),
            "## 5. Information Gap Analysis\n\nThe key information gap is whether the observed signal is merely a news item or an actionable opportunity for the user.",
            "## 6. Opportunities\n\n" + _bullets(opportunities, "title", "detail"),
            "## 7. Risks and Uncertainties\n\n" + _bullets(risks, "title", "detail"),
            "## 8. Suggested Actions\n\n" + _bullets(actions, "title", "detail"),
            "## 9. Sources\n\n" + _sources(evidence),
        ]
    )


def _bullets(rows: list[dict[str, Any]], title_key: str, detail_key: str) -> str:
    if not rows:
        return "- Evidence is insufficient; treat conclusions as low confidence."
    return "\n".join(f"- **{row.get(title_key, 'Item')}**: {row.get(detail_key, row.get('description', ''))}" for row in rows)


def _evidence(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- No evidence found. This report is evidence_insufficient."
    return "\n".join(f"- **{row.get('title', 'Source')}** ({row.get('source_type', 'unknown')}, score={row.get('score', 0)}): {row.get('snippet', '')}" for row in rows)


def _sources(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- No sources available."
    return "\n".join(f"- {row.get('title', 'Source')}: {row.get('url') or 'no URL'}" for row in rows)
