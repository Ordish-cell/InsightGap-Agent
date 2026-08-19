from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

LLMProtocol = Literal[
    "openai_chat_completions",
    "openai_responses",
    "anthropic_messages",
    "google_generate_content",
    "ollama_chat",
]


@dataclass(frozen=True)
class ProviderField:
    key: str
    label: str
    kind: Literal["text", "secret", "secret_json", "select", "url", "json"] = "text"
    required: bool = False
    default: Any = ""
    options: tuple[dict[str, str], ...] = ()
    placeholder: str = ""


@dataclass(frozen=True)
class PresetModel:
    model_id: str
    display_name: str
    tier: Literal["strong", "balanced", "fast"]
    capabilities: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderDefinition:
    key: str
    label: str
    protocol: LLMProtocol
    fields: tuple[ProviderField, ...]
    capabilities: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_CAPABILITIES))
    models: tuple[PresetModel, ...] = ()
    discovery: str = "openai_models"
    protocols: tuple[LLMProtocol, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["protocols"] = list(self.protocols or (self.protocol,))
        return payload


DEFAULT_CAPABILITIES = {"tools": True, "structured_output": True, "streaming": True}


def _field(key: str, label: str, *, kind: str = "text", required: bool = False, default: Any = "", options: tuple[dict[str, str], ...] = (), placeholder: str = "") -> ProviderField:
    return ProviderField(key=key, label=label, kind=kind, required=required, default=default, options=options, placeholder=placeholder)  # type: ignore[arg-type]


def _models(*items: tuple[str, str, str]) -> tuple[PresetModel, ...]:
    return tuple(PresetModel(model_id=model_id, display_name=name, tier=tier, capabilities=dict(DEFAULT_CAPABILITIES)) for model_id, name, tier in items)  # type: ignore[arg-type]


PROVIDERS: dict[str, ProviderDefinition] = {
    "openai": ProviderDefinition(
        key="openai", label="OpenAI", protocol="openai_responses", discovery="openai_models",
        fields=(
            _field("api_key", "API Key", kind="secret", required=True, placeholder="sk-…"),
            _field("base_url", "Base URL", kind="url", default="https://api.openai.com/v1"),
        ),
        models=_models(("gpt-5.4", "GPT-5.4", "strong"), ("gpt-5.4-mini", "GPT-5.4 mini", "balanced"), ("gpt-5.4-nano", "GPT-5.4 nano", "fast")),
        protocols=("openai_responses", "openai_chat_completions"),
    ),
    "azure_openai": ProviderDefinition(
        key="azure_openai", label="Azure OpenAI", protocol="openai_chat_completions", discovery="none",
        fields=(
            _field("api_key", "API Key", kind="secret", required=True),
            _field("endpoint", "Azure Endpoint", kind="url", required=True, placeholder="https://….openai.azure.com"),
            _field("deployment", "Deployment", required=True),
            _field("api_version", "API Version", required=True, default="2025-04-01-preview"),
        ),
    ),
    "anthropic": ProviderDefinition(
        key="anthropic", label="Anthropic", protocol="anthropic_messages", discovery="anthropic_models",
        fields=(
            _field("api_key", "API Key", kind="secret", required=True, placeholder="sk-ant-…"),
            _field("base_url", "Base URL", kind="url", default="https://api.anthropic.com"),
        ),
        models=_models(("claude-opus-4-6", "Claude Opus 4.6", "strong"), ("claude-sonnet-4-6", "Claude Sonnet 4.6", "balanced"), ("claude-haiku-4-5", "Claude Haiku 4.5", "fast")),
    ),
    "gemini": ProviderDefinition(
        key="gemini", label="Google Gemini", protocol="google_generate_content", discovery="gemini_models",
        fields=(_field("api_key", "API Key", kind="secret", required=True),),
        models=_models(("gemini-3.1-pro", "Gemini 3.1 Pro", "strong"), ("gemini-3.6-flash", "Gemini 3.6 Flash", "balanced"), ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "fast")),
    ),
    "deepseek": ProviderDefinition(
        key="deepseek", label="DeepSeek", protocol="openai_chat_completions", discovery="openai_models",
        fields=(
            _field("api_key", "API Key", kind="secret", required=True),
            _field("base_url", "Base URL", kind="url", default="https://api.deepseek.com"),
        ),
        models=_models(("deepseek-v4-pro", "DeepSeek V4 Pro", "strong"), ("deepseek-v4-flash", "DeepSeek V4 Flash", "fast")),
    ),
    "qwen": ProviderDefinition(
        key="qwen", label="通义千问", protocol="openai_responses", discovery="openai_models",
        fields=(
            _field("api_key", "API Key", kind="secret", required=True),
            _field("region", "地域", kind="select", required=True, default="cn", options=({"label": "中国大陆", "value": "cn"}, {"label": "国际", "value": "intl"}, {"label": "自定义", "value": "custom"})),
            _field("base_url", "Base URL", kind="url", default="https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ),
        models=_models(("qwen3.7-max", "Qwen 3.7 Max", "strong"), ("qwen3.7-plus", "Qwen 3.7 Plus", "balanced"), ("qwen3.6-flash", "Qwen 3.6 Flash", "fast")),
        protocols=("openai_responses", "openai_chat_completions"),
    ),
    "openrouter": ProviderDefinition(
        key="openrouter", label="OpenRouter", protocol="openai_chat_completions", discovery="openrouter_models",
        fields=(
            _field("api_key", "API Key", kind="secret", required=True),
            _field("base_url", "Base URL", kind="url", default="https://openrouter.ai/api/v1"),
            _field("site_url", "站点 URL", kind="url"),
            _field("app_name", "应用名称"),
        ),
    ),
    "ollama": ProviderDefinition(
        key="ollama", label="Ollama", protocol="ollama_chat", discovery="ollama_tags",
        fields=(_field("base_url", "Base URL", kind="url", required=True, default="http://127.0.0.1:11434"),),
    ),
    "custom": ProviderDefinition(
        key="custom", label="自定义模型", protocol="openai_chat_completions", discovery="none",
        fields=(
            _field("base_url", "Base URL 或完整 Endpoint", kind="url", required=True),
            _field("api_key", "API Key", kind="secret"),
            _field("auth_header", "认证 Header", default="Authorization"),
            _field("custom_headers", "自定义 Headers", kind="secret_json", default={}),
        ),
        protocols=("openai_chat_completions", "openai_responses", "anthropic_messages", "google_generate_content", "ollama_chat"),
    ),
}


SECRET_FIELD_KEYS = {field.key for provider in PROVIDERS.values() for field in provider.fields if field.kind in {"secret", "secret_json"}}


def get_provider(key: str) -> ProviderDefinition:
    try:
        return PROVIDERS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM provider: {key}") from exc


def catalog() -> list[dict[str, Any]]:
    return [provider.public_dict() for provider in PROVIDERS.values()]
