from src.web_app.agent.llm.errors import LLMUnavailableError
from src.web_app.agent.llm.router import resolve_model_name
from src.web_app.core.config import get_settings


def get_embedding_model() -> dict[str, str]:
    settings = get_settings()
    resolution = resolve_model_name("embedding", complexity="low")
    if not resolution.model:
        raise LLMUnavailableError("Embedding model is not configured")
    provider = settings.embed_model_type or resolution.provider
    if not provider:
        raise LLMUnavailableError("Embedding provider is not configured")
    return {"provider": provider, "model": resolution.model, "tier": resolution.tier}
