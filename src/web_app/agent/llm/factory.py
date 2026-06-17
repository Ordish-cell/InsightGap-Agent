from functools import lru_cache
from typing import Any

from src.web_app.agent.llm.config import get_llm_settings
from src.web_app.agent.llm.errors import LLMUnavailableError
from src.web_app.agent.llm.router import ModelPurpose, resolve_model_name


def get_chat_model(purpose: ModelPurpose | str, complexity: str = "normal", temperature: float | None = None, streaming: bool = False) -> Any:
    settings = get_llm_settings()
    resolution = resolve_model_name(purpose, complexity)
    return _cached_chat_model(
        provider=resolution.provider,
        model=resolution.model,
        base_url=settings.effective_base_url,
        api_key=settings.effective_api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        temperature=settings.temperature if temperature is None else temperature,
        enabled=settings.enabled,
        streaming=streaming,
    )


def get_chat_model_by_name(model: str, temperature: float | None = None, timeout_seconds: int | None = None, streaming: bool = False) -> Any:
    """Build a chat model with the existing provider/API-key configuration."""
    settings = get_llm_settings()
    return _cached_chat_model(
        provider=settings.provider,
        model=model,
        base_url=settings.effective_base_url,
        api_key=settings.effective_api_key,
        timeout=settings.timeout_seconds if timeout_seconds is None else timeout_seconds,
        max_retries=settings.max_retries,
        temperature=settings.temperature if temperature is None else temperature,
        enabled=settings.enabled,
        streaming=streaming,
    )


@lru_cache
def _cached_chat_model(provider: str, model: str, base_url: str, api_key: str, timeout: int, max_retries: int, temperature: float, enabled: bool, streaming: bool = False) -> Any:
    if not enabled or provider == "disabled":
        raise LLMUnavailableError("Agent LLM is disabled")
    if provider not in {"aliyun", "openai_compatible"}:
        raise LLMUnavailableError(f"Unsupported LLM provider: {provider}")
    if not api_key:
        raise LLMUnavailableError(f"LLM API key is not configured for provider={provider}")
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise LLMUnavailableError(f"langchain_openai is unavailable: {exc}") from exc
    return ChatOpenAI(
        model=model,
        base_url=base_url or None,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        streaming=streaming,
    )


def clear_chat_model_cache() -> None:
    _cached_chat_model.cache_clear()
