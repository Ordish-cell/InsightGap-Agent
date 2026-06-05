from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["aliyun", "openai_compatible", "disabled"]


class LLMSettings(BaseSettings):
    enabled: bool = Field(True, validation_alias="AGENT_LLM_ENABLED")
    provider: LLMProvider = Field("aliyun", validation_alias="AGENT_LLM_PROVIDER")
    timeout_seconds: int = Field(60, validation_alias="AGENT_LLM_TIMEOUT_SECONDS")
    max_retries: int = Field(2, validation_alias="AGENT_LLM_MAX_RETRIES")
    temperature: float = Field(0.2, validation_alias="AGENT_LLM_TEMPERATURE")
    dashscope_api_key: str = Field("", validation_alias="DASHSCOPE_API_KEY")
    aliyun_bailian_api_key: str = Field("", validation_alias="ALIYUN_BAILIAN_API_KEY")
    aliyun_bailian_base_url: str = Field("https://dashscope.aliyuncs.com/compatible-mode/v1", validation_alias="ALIYUN_BAILIAN_BASE_URL")
    base_url: str = Field("", validation_alias="AGENT_LLM_BASE_URL")
    api_key: str = Field("", validation_alias="AGENT_LLM_API_KEY")
    fast_model: str = Field("qwen3.6-flash", validation_alias="AGENT_FAST_MODEL")
    balanced_model: str = Field("qwen3.6-max-preview", validation_alias="AGENT_BALANCED_MODEL")
    strong_model: str = Field("qwen3.7-plus", validation_alias="AGENT_STRONG_MODEL")
    default_model: str = Field("qwen3.6-plus", validation_alias="AGENT_LLM_MODEL")
    intent_model: str = Field("qwen3.6-flash", validation_alias="AGENT_INTENT_MODEL")
    safety_model: str = Field("qwen3.6-flash", validation_alias="AGENT_SAFETY_MODEL")
    planner_model: str = Field("qwen3.6-max-preview", validation_alias="AGENT_PLANNER_MODEL")
    rag_model: str = Field("qwen3.6-max-preview", validation_alias="AGENT_RAG_MODEL")
    research_model: str = Field("qwen3.7-plus", validation_alias="AGENT_RESEARCH_MODEL")
    artifact_model: str = Field("qwen3.6-plus", validation_alias="AGENT_ARTIFACT_MODEL")
    memory_model: str = Field("qwen3.6-flash", validation_alias="AGENT_MEMORY_MODEL")
    skill_model: str = Field("qwen3.6-flash", validation_alias="AGENT_SKILL_MODEL")
    final_model: str = Field("qwen3.6-plus", validation_alias="AGENT_FINAL_MODEL")
    embedding_provider: str = Field("aliyun", validation_alias="AGENT_EMBEDDING_PROVIDER")
    embedding_model: str = Field("text-embedding-v4", validation_alias="AGENT_EMBEDDING_MODEL")
    intent_llm_enabled: bool = Field(True, validation_alias="AGENT_INTENT_LLM_ENABLED")
    planner_llm_enabled: bool = Field(False, validation_alias="AGENT_PLANNER_LLM_ENABLED")
    memory_llm_enabled: bool = Field(True, validation_alias="AGENT_MEMORY_LLM_ENABLED")
    skill_llm_enabled: bool = Field(True, validation_alias="AGENT_SKILL_LLM_ENABLED")
    safety_llm_enabled: bool = Field(True, validation_alias="AGENT_SAFETY_LLM_ENABLED")
    usage_log_enabled: bool = Field(True, validation_alias="AGENT_LLM_USAGE_LOG_ENABLED")
    log_prompt_preview: bool = Field(False, validation_alias="AGENT_LLM_LOG_PROMPT_PREVIEW")
    log_raw_output: bool = Field(False, validation_alias="AGENT_LLM_LOG_RAW_OUTPUT")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def effective_api_key(self) -> str:
        if self.provider == "aliyun":
            return self.dashscope_api_key or self.aliyun_bailian_api_key
        if self.provider == "openai_compatible":
            return self.api_key
        return ""

    @property
    def effective_base_url(self) -> str:
        if self.provider == "aliyun":
            return self.aliyun_bailian_base_url
        return self.base_url


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


def clear_llm_settings_cache() -> None:
    get_llm_settings.cache_clear()
