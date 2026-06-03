def should_promote(memory_type: str, importance: float) -> bool:
    return (memory_type == "working" and importance >= 0.7) or (memory_type == "episodic" and importance >= 0.8)
