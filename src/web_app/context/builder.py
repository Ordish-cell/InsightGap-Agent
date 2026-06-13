from typing import Any

from src.web_app.context.packets import ContextConfig, ContextPacket


SECTIONS = [
    "Role & Policies",
    "User Profile",
    "Task",
    "State",
    "Conversation History",
    "Relevant Memory",
    "Evidence",
    "Information Gap Signals",
    "Tool State",
    "Output Contract",
    "Conversation Summary",
    "Checkpoint Summary",
    "Feed Card Context",
    "Dynamic Preferences",
    "Graph Context",
]

SOURCE_MAP = {
    "profile": "User Profile",
    "task": "Task",
    "state": "State",
    "conversation_history": "Conversation History",
    "memory": "Relevant Memory",
    "evidence": "Evidence",
    "information_gap": "Information Gap Signals",
    "tool_state": "Tool State",
    "output_contract": "Output Contract",
    "conversation_summary": "Conversation Summary",
    "checkpoint_summary": "Checkpoint Summary",
    "feed_card": "Feed Card Context",
    "dynamic_preferences": "Dynamic Preferences",
    "graph_context": "Graph Context",
}

# Route-adaptive source relevance weights.
# Higher → more likely to survive Select phase within token budget.
_ROUTE_WEIGHTS: dict[str, dict[str, float]] = {
    "chat": {
        "conversation_history": 0.95, "conversation_summary": 0.80,
        "memory": 0.75, "profile": 0.60, "task": 0.70,
        "feed_card": 0.50, "evidence": 0.35, "checkpoint_summary": 0.40,
        "dynamic_preferences": 0.65, "graph_context": 0.62,
    },
    "feed": {
        "memory": 0.85, "dynamic_preferences": 0.80,
        "conversation_history": 0.75, "profile": 0.65,
        "conversation_summary": 0.30, "evidence": 0.45,
        "feed_card": 0.55, "task": 0.60, "graph_context": 0.55,
    },
    "research": {
        "feed_card": 0.85, "evidence": 0.80,
        "conversation_history": 0.75, "checkpoint_summary": 0.70,
        "memory": 0.65, "dynamic_preferences": 0.60,
        "task": 0.75, "conversation_summary": 0.55,
        "profile": 0.50, "graph_context": 0.60,
    },
    "rag": {
        "evidence": 0.90, "task": 0.75,
        "conversation_history": 0.70, "feed_card": 0.65,
        "memory": 0.55, "checkpoint_summary": 0.40, "graph_context": 0.50,
    },
    "skill": {
        "memory": 0.80, "conversation_history": 0.80,
        "dynamic_preferences": 0.75, "task": 0.70,
        "conversation_summary": 0.65, "feed_card": 0.55,
        "evidence": 0.40, "checkpoint_summary": 0.50, "graph_context": 0.70,
    },
    "artifact": {
        "dynamic_preferences": 0.85, "conversation_history": 0.80,
        "feed_card": 0.75, "task": 0.70, "memory": 0.65,
        "evidence": 0.60, "conversation_summary": 0.55,
        "checkpoint_summary": 0.50, "graph_context": 0.65,
    },
    "tool": {
        "task": 0.80, "conversation_history": 0.75,
        "evidence": 0.70, "checkpoint_summary": 0.65,
        "memory": 0.45, "feed_card": 0.40, "graph_context": 0.55,
    },
}

# ── Memory context policy ────────────────────────────────────────────
# Controls which memory *categories* are injected into context per answer_mode.
# This prevents irrelevant memories (e.g. tech_stack) from polluting casual chat.
# Key constraint: general_qa must NOT inject project_goal/tech_stack/boundary/workflow_pattern
#                 memory_confirm allows name/tone but NOT tech_stack/project_goal
MEMORY_CONTEXT_POLICY: dict[str, set[str]] = {
    "memory_confirm": {"name_preference", "language_preference", "tone_preference"},
    "casual": {"name_preference", "language_preference", "tone_preference"},
    "general_qa": {"name_preference", "language_preference", "answer_preference", "tone_preference"},
    "rag_qa": {"name_preference", "language_preference", "answer_preference", "document_preference"},
    "tool_action": {"name_preference", "language_preference", "tool_preference", "boundary"},
    "project_advice": {
        "name_preference", "language_preference", "answer_preference",
        "project_goal", "tech_stack", "boundary", "workflow_pattern",
        "preference", "negative_preference",
    },
    "chat": {
        "name_preference", "language_preference", "answer_preference",
        "tone_preference", "project_goal", "tech_stack", "boundary",
        "workflow_pattern", "feed_interest", "research_preference",
    },
}


class ContextBuilder:
    def __init__(self, config: ContextConfig | None = None, route: str = "chat"):
        self.config = config or ContextConfig()
        self.route = route
        self._selected_sources: list[str] = []
        self._dropped_sources: list[str] = []
        self._token_budget_used = 0

    def gather(self, payload: dict[str, Any]) -> list[ContextPacket]:
        packets: list[ContextPacket] = []
        weights = _ROUTE_WEIGHTS.get(self.route, _ROUTE_WEIGHTS["chat"])
        for source, content in payload.items():
            if content in (None, "", [], {}):
                continue
            text = content if isinstance(content, str) else str(content)
            # Route-aware relevance: use route-specific weight if defined,
            # fall back to default heuristics
            relevance = weights.get(source, 0.35)
            if source == "evidence" and isinstance(content, list) and content:
                relevance = max(float(item.get("score", relevance)) for item in content if isinstance(item, dict))
            packets.append(
                ContextPacket(
                    content=text,
                    token_count=max(1, len(text) // 4),
                    relevance_score=relevance,
                    metadata={"source": source},
                )
            )
        return packets

    def select(self, packets: list[ContextPacket]) -> list[ContextPacket]:
        budget = int(self.config.max_tokens * (1 - self.config.reserve_ratio))
        selected: list[ContextPacket] = []
        used = 0
        self._selected_sources = []
        self._dropped_sources = []

        for packet in sorted(packets, key=lambda item: item.relevance_score, reverse=True):
            if packet.relevance_score < self.config.min_relevance:
                self._dropped_sources.append(packet.metadata.get("source", "?"))
                continue
            if used + packet.token_count > budget:
                self._dropped_sources.append(packet.metadata.get("source", "?"))
                continue
            selected.append(packet)
            used += packet.token_count
            self._selected_sources.append(packet.metadata.get("source", "?"))

        self._token_budget_used = used
        return selected

    def structure(self, packets: list[ContextPacket]) -> str:
        grouped = {section: [] for section in SECTIONS}
        grouped["Role & Policies"].append(
            "Use user-scoped evidence, require approvals for external writes, "
            "and keep outputs traceable."
        )
        for packet in packets:
            section = SOURCE_MAP.get(packet.metadata.get("source"), "State")
            if section in grouped:
                grouped[section].append(packet.content)
        return "\n\n".join(
            f"[{section}]\n" + "\n".join(items)
            for section, items in grouped.items() if items
        )

    def compress(self, context: str) -> str:
        max_chars = self.config.max_tokens * 4
        if not self.config.enable_compression or len(context) <= max_chars:
            return context
        # Smart compress: drop the lowest-priority sections first
        return context[: max_chars - 32] + "\n[compressed]"

    def build(self, payload: dict[str, Any]) -> str:
        return self.compress(self.structure(self.select(self.gather(payload))))

    def build_with_debug(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        context = self.build(payload)
        debug = {
            "gssc_route": self.route,
            "selected_sources": self._selected_sources,
            "dropped_sources": self._dropped_sources,
            "token_budget_used": self._token_budget_used,
            "token_budget_max": self.config.max_tokens,
        }
        return context, debug

    def evaluate_context_quality(self, packets: list[ContextPacket]) -> dict[str, Any]:
        token_usage = sum(packet.token_count for packet in packets)
        relevance_avg = sum(packet.relevance_score for packet in packets) / len(packets) if packets else 0.0
        evidence_count = sum(1 for packet in packets if packet.metadata.get("source") == "evidence")
        memory_count = sum(1 for packet in packets if packet.metadata.get("source") == "memory")
        risk_flags = ["low_evidence"] if evidence_count == 0 else []
        return {
            "token_usage": token_usage,
            "relevance_avg": round(relevance_avg, 4),
            "evidence_count": evidence_count,
            "memory_count": memory_count,
            "risk_flags": risk_flags,
            "quality_score": round(min(1.0, relevance_avg + evidence_count * 0.1 + memory_count * 0.05), 4),
            "route": self.route,
            "selected_sources": self._selected_sources,
        }
