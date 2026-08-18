from dataclasses import asdict, dataclass
from typing import Literal

from src.web_app.agent.llm.context import get_model_context
from src.web_app.agent.llm.errors import LLMRouterError
from src.web_app.core.config import get_settings

ModelPurpose = Literal[
    "intent",
    "safety",
    "planner",
    "rag",
    "research",
    "artifact",
    "memory",
    "skill",
    "final",
    "embedding",
    "default",
]
ModelTier = Literal["fast", "balanced", "strong"]


@dataclass(frozen=True)
class ModelResolution:
    purpose: ModelPurpose
    tier: ModelTier
    provider: str
    model: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_TIER_BY_PURPOSE: dict[ModelPurpose, ModelTier] = {
    "intent": "fast",
    "safety": "fast",
    "memory": "fast",
    "skill": "fast",
    "planner": "balanced",
    "rag": "balanced",
    "artifact": "balanced",
    "final": "balanced",
    "research": "strong",
    "embedding": "fast",
    "default": "balanced",
}


def resolve_model_name(purpose: ModelPurpose | str, complexity: str = "normal") -> ModelResolution:
    normalized = _normalize_purpose(purpose)
    tier = _resolve_tier(normalized, complexity)
    if normalized == "embedding":
        settings = get_settings()
        model = settings.embed_model_name
        provider = settings.embed_model_type
    else:
        context = get_model_context()
        model = context.model
        provider = context.provider
    if not model:
        raise LLMRouterError(f"No model configured for purpose={normalized}")
    return ModelResolution(purpose=normalized, tier=tier, provider=provider, model=model)


def _normalize_purpose(purpose: ModelPurpose | str) -> ModelPurpose:
    allowed = set(DEFAULT_TIER_BY_PURPOSE)
    if purpose in allowed:
        return purpose  # type: ignore[return-value]
    return "default"


def _resolve_tier(purpose: ModelPurpose, complexity: str) -> ModelTier:
    if complexity == "high" and purpose not in {"intent", "safety", "memory", "skill", "embedding"}:
        return "strong"
    if complexity == "low":
        return "fast"
    return DEFAULT_TIER_BY_PURPOSE[purpose]
