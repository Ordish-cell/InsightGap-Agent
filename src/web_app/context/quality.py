def quality_score(relevance_avg: float, evidence_count: int, memory_count: int) -> float:
    return round(min(1.0, relevance_avg + evidence_count * 0.1 + memory_count * 0.05), 4)
