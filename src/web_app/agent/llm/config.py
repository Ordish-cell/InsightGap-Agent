from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    enabled: bool = Field(True, validation_alias="LLM_ENABLED")
    intent_llm_enabled: bool = Field(True, validation_alias="AGENT_INTENT_LLM_ENABLED")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


def clear_llm_settings_cache() -> None:
    get_llm_settings.cache_clear()
