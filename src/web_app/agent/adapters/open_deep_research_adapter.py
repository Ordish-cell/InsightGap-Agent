from typing import Any


class OpenDeepResearchAdapter:
    def run_research(self, query: str, context: Any, user_id: int, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": f"Mock research result for: {query}",
            "findings": [],
            "evidence": [],
            "risks": [],
            "opportunities": [],
            "suggested_actions": [],
            "user_id": user_id,
            "mode": config.get("mode", "mock"),
        }
