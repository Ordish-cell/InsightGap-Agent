from __future__ import annotations

from typing import Any

from src.web_app.agent.llm.context import ModelExecutionContext, get_model_context
from src.web_app.agent.llm.errors import LLMUnavailableError
from src.web_app.agent.llm.router import ModelPurpose
from src.web_app.core.config import get_settings


def get_chat_model(purpose: ModelPurpose | str, complexity: str = "normal", temperature: float | None = None, streaming: bool = False) -> Any:
    del purpose, complexity
    return build_chat_model(get_model_context(), temperature=temperature, streaming=streaming)


def get_chat_model_by_name(model: str, temperature: float | None = None, timeout_seconds: int | None = None, streaming: bool = False) -> Any:
    context = get_model_context()
    del model
    return build_chat_model(context, temperature=temperature, timeout_seconds=timeout_seconds, streaming=streaming)


def build_chat_model(context: ModelExecutionContext, *, temperature: float | None = 0.2, timeout_seconds: int | None = None, streaming: bool = False) -> Any:
    settings = get_settings()
    timeout = timeout_seconds or settings.llm_timeout_seconds
    retries = settings.llm_max_retries
    protocol = context.protocol
    config = context.config
    api_key = str(context.secrets.get("api_key") or "")
    base_url = normalize_model_endpoint(str(config.get("base_url") or ""), protocol)
    headers = dict(config.get("custom_headers") or {})
    if context.provider == "custom" and api_key and config.get("auth_header") not in (None, "", "Authorization"):
        headers[str(config["auth_header"])] = api_key
        api_key = "not-required"
    common: dict[str, Any] = {"model": context.model, "timeout": timeout, "max_retries": retries, "streaming": streaming}
    if temperature is not None and _supports_temperature(context.model):
        common["temperature"] = temperature
    try:
        if context.provider == "azure_openai":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_endpoint=config.get("endpoint"), api_key=api_key,
                azure_deployment=config.get("deployment") or context.model,
                api_version=config.get("api_version"), **{key: value for key, value in common.items() if key != "model"},
            )
        if protocol == "anthropic_messages":
            from langchain_anthropic import ChatAnthropic
            kwargs = {**common, "api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            if headers:
                kwargs["default_headers"] = headers
            return ChatAnthropic(**kwargs)
        if protocol == "google_generate_content":
            from langchain_google_genai import ChatGoogleGenerativeAI
            kwargs = {**common, "google_api_key": api_key}
            if context.provider == "custom" and base_url:
                kwargs["base_url"] = base_url
            if headers:
                kwargs["additional_headers"] = headers
            return ChatGoogleGenerativeAI(**kwargs)
        from langchain_openai import ChatOpenAI
        if protocol == "ollama_chat":
            base_url = f"{base_url}/v1" if not base_url.endswith("/v1") else base_url
            api_key = api_key or "ollama"
        if context.provider == "openrouter":
            if config.get("site_url"):
                headers["HTTP-Referer"] = config["site_url"]
            if config.get("app_name"):
                headers["X-OpenRouter-Title"] = config["app_name"]
        kwargs = {**common, "api_key": api_key or "not-required", "base_url": base_url or None}
        if headers:
            kwargs["default_headers"] = headers
        if protocol == "openai_responses":
            kwargs["use_responses_api"] = True
        return ChatOpenAI(**kwargs)
    except Exception as exc:
        raise LLMUnavailableError(f"Failed to initialize {context.provider}/{context.model}: {exc}") from exc


def clear_chat_model_cache() -> None:
    return None


def _supports_temperature(model: str) -> bool:
    normalized = model.lower()
    return not normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def normalize_model_endpoint(value: str, protocol: str) -> str:
    base_url = value.rstrip("/")
    suffixes = {
        "openai_chat_completions": ("/chat/completions",),
        "openai_responses": ("/responses",),
        "anthropic_messages": ("/v1/messages", "/messages"),
        "ollama_chat": ("/api/chat", "/chat/completions"),
    }.get(protocol, ())
    for suffix in suffixes:
        if base_url.endswith(suffix):
            return base_url[:-len(suffix)]
    if protocol == "google_generate_content" and "/models/" in base_url:
        return base_url.split("/models/", 1)[0]
    return base_url
