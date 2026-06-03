from typing import Any

from src.web_app.context.packets import ContextConfig, ContextPacket


SECTIONS = [
    "Role & Policies",
    "User Profile",
    "Task",
    "State",
    "Relevant Memory",
    "Evidence",
    "Information Gap Signals",
    "Tool State",
    "Output Contract",
]


class ContextBuilder:
    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()

    def gather(self, payload: dict[str, Any]) -> list[ContextPacket]:
        packets: list[ContextPacket] = []
        for source, content in payload.items():
            if content in (None, "", [], {}):
                continue
            text = content if isinstance(content, str) else str(content)
            relevance = 0.5 if source in {"task", "evidence", "memory"} else 0.3
            if source == "evidence" and isinstance(content, list) and content:
                relevance = max(float(item.get("score", 0.5)) for item in content if isinstance(item, dict))
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
        for packet in sorted(packets, key=lambda item: item.relevance_score, reverse=True):
            if packet.relevance_score < self.config.min_relevance:
                continue
            if used + packet.token_count > budget:
                continue
            selected.append(packet)
            used += packet.token_count
        return selected

    def structure(self, packets: list[ContextPacket]) -> str:
        grouped = {section: [] for section in SECTIONS}
        source_map = {
            "profile": "User Profile",
            "task": "Task",
            "state": "State",
            "memory": "Relevant Memory",
            "evidence": "Evidence",
            "information_gap": "Information Gap Signals",
            "tool_state": "Tool State",
            "output_contract": "Output Contract",
        }
        grouped["Role & Policies"].append("Use user-scoped evidence, require approvals for external writes, and keep outputs traceable.")
        for packet in packets:
            section = source_map.get(packet.metadata.get("source"), "State")
            grouped[section].append(packet.content)
        return "\n\n".join(f"[{section}]\n" + "\n".join(items) for section, items in grouped.items())

    def compress(self, context: str) -> str:
        max_chars = self.config.max_tokens * 4
        if not self.config.enable_compression or len(context) <= max_chars:
            return context
        return context[: max_chars - 32] + "\n[compressed]"

    def build(self, payload: dict[str, Any]) -> str:
        return self.compress(self.structure(self.select(self.gather(payload))))

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
        }
