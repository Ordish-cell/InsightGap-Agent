from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ContextPacket(BaseModel):
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token_count: int = 0
    relevance_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextConfig(BaseModel):
    max_tokens: int = 4000
    reserve_ratio: float = 0.20
    min_relevance: float = 0.10
    enable_compression: bool = True
    recency_weight: float = 0.30
    relevance_weight: float = 0.70
