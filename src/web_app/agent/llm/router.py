from dataclasses import asdict, dataclass
from typing import Literal

from src.web_app.agent.llm.config import get_llm_settings
from src.web_app.agent.llm.errors import LLMRouterError

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
    settings = get_llm_settings()
    normalized = _normalize_purpose(purpose)
    tier = _resolve_tier(normalized, complexity)
    if normalized == "embedding":
        model = settings.embedding_model
        provider = settings.embedding_provider
    else:
        model = _purpose_model(normalized) or _tier_model(tier) or settings.default_model
        provider = settings.provider
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


def _purpose_model(purpose: ModelPurpose) -> str:
    settings = get_llm_settings()
    return {
        "intent": settings.intent_model,
        "safety": settings.safety_model,
        "planner": settings.planner_model,
        "rag": settings.rag_model,
        "research": settings.research_model,
        "artifact": settings.artifact_model,
        "memory": settings.memory_model,
        "skill": settings.skill_model,
        "final": settings.final_model,
    }.get(purpose, "")


def _tier_model(tier: ModelTier) -> str:
    settings = get_llm_settings()
    return {
        "fast": settings.fast_model,
        "balanced": settings.balanced_model,
        "strong": settings.strong_model,
    }[tier]
