from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 10080
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "open_deep_research"
    redis_url: str = "redis://127.0.0.1:6379/0"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "agent_os_documents"
    qdrant_vector_size: int = 384
    qdrant_distance: str = "cosine"
    qdrant_timeout: int = 30
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_max_connection_lifetime: int = 3600
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 60
    enable_neo4j: bool = False
    embed_model_type: str = "dashscope"
    embed_model_name: str = "text-embedding-v4"
    embed_api_key: str = ""
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_llm_provider: str = "openai"
    default_llm_model: str = ""
    embedding_provider: str = "openai"
    embedding_model: str = ""
    open_deep_research_mode: str = "adapter"
    artifact_storage_path: str = "uploads/artifacts"
    cors_origins_raw: str = Field("http://localhost:5173,http://127.0.0.1:5173", validation_alias="CORS_ORIGINS")
    feed_refresh_max_items: int = 50
    feed_card_limit_default: int = 20
    feed_manual_seed_enabled: bool = True
    feed_github_enabled: bool = True
    feed_github_topics: str = "agent,rag,llm,langgraph,mcp"
    feed_github_languages: str = "python,typescript"
    feed_github_max_items: int = 20
    feed_github_min_stars: int = 50
    feed_github_pushed_days: int = 90
    github_token: str = ""
    feed_arxiv_enabled: bool = True
    feed_arxiv_categories: str = "cs.AI,cs.CL,cs.LG"
    feed_arxiv_queries: str = "agent,rag,multi-agent,tool use,llm,browser agent"
    feed_arxiv_max_items: int = 20
    feed_rss_enabled: bool = True
    feed_rss_urls: str = ""
    feed_rss_max_items: int = 20
    feed_tavily_enabled: bool = True
    tavily_api_key: str = ""
    feed_tavily_max_items: int = 10
    feed_tavily_search_depth: str = "basic"
    feed_tavily_include_answer: bool = False
    feed_tavily_include_raw_content: bool = False
    feed_serpapi_enabled: bool = True
    serpapi_api_key: str = ""
    feed_serpapi_engine: str = "google"
    feed_serpapi_max_items: int = 10
    feed_serpapi_location: str = ""
    feed_serpapi_hl: str = "zh-cn"
    feed_serpapi_gl: str = "cn"
    feed_duckduckgo_enabled: bool = True
    feed_duckduckgo_max_items: int = 10
    feed_duckduckgo_region: str = "wt-wt"
    feed_duckduckgo_safesearch: str = "moderate"
    feed_duckduckgo_time: str = "w"
    feed_ratio_explicit: float = 0.30
    feed_ratio_adjacent: float = 0.40
    feed_ratio_far: float = 0.30
    feed_min_personal_relevance: float = 0.15
    feed_min_source_credibility: float = 0.40
    feed_low_confidence_max_ratio: float = 0.20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def mysql_url(self) -> str:
        password = self.mysql_password
        return (
            f"mysql+pymysql://{self.mysql_user}:{password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]

    def csv(self, value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
