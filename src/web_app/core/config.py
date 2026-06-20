from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 10080
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_database: str = "agent_os"
    redis_url: str = "redis://127.0.0.1:6379/0"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "agent_os_documents"
    qdrant_vector_size: int = 384
    qdrant_distance: str = "cosine"
    qdrant_timeout: int = 30
    rag_hybrid_backend: str = "python_bm25"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "bm25"
    qdrant_fusion_method: str = "rrf"
    qdrant_hybrid_fallback: bool = True
    qdrant_hybrid_collection: str = "agent_os_documents_v3"
    qdrant_sparse_encoder: str = "qdrant_cloud_bm25"
    qdrant_sparse_model: str = "Qdrant/bm25"
    qdrant_cloud_inference: bool = True
    qdrant_sparse_hash_size: int = 2_000_003
    memory_qdrant_collection: str = "memory_vectors"
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_max_connection_lifetime: int = 3600
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 60
    enable_neo4j: bool = False
    neo4j_memory_graph_enabled: bool = True
    neo4j_project_graph_enabled: bool = True
    neo4j_context_enabled: bool = True
    neo4j_write_mode: str = "best_effort"
    neo4j_project_key: str = "agent_os"
    neo4j_graph_context_limit: int = 8
    embed_model_type: str = "dashscope"
    embed_model_name: str = "text-embedding-v4"
    embed_api_key: str = ""
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_embedding_batch_size: int = 10
    default_llm_provider: str = "openai"
    default_llm_model: str = ""
    embedding_provider: str = "openai"
    embedding_model: str = ""
    open_deep_research_mode: str = "adapter"
    # ── Open Deep Research (upstream) ──────────────────────────────
    enable_open_deep_research: bool = True
    odr_search_api: str = "tavily"
    odr_allow_clarification: bool = False
    odr_max_concurrent_research_units: int = 2
    odr_max_researcher_iterations: int = 2
    odr_max_react_tool_calls: int = 4
    odr_timeout_seconds: int = 600
    odr_research_model: str = "openai:qwen-plus"
    odr_summarization_model: str = "openai:qwen-plus"
    odr_compression_model: str = "openai:qwen-plus"
    odr_final_report_model: str = "openai:qwen-plus"
    odr_research_model_max_tokens: int = 10000
    odr_summarization_model_max_tokens: int = 8192
    odr_compression_model_max_tokens: int = 8192
    odr_final_report_model_max_tokens: int = 10000
    odr_max_content_length: int = 50000
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
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
    agent_llm_enabled: bool = True
    agent_llm_provider: str = "aliyun"
    agent_llm_timeout_seconds: int = 60
    agent_llm_max_retries: int = 2
    agent_llm_temperature: float = 0.2
    dashscope_api_key: str = ""
    aliyun_bailian_api_key: str = ""
    aliyun_bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    agent_llm_base_url: str = ""
    agent_llm_api_key: str = ""
    agent_fast_model: str = "qwen3.6-max-preview"
    agent_balanced_model: str = "qwen3.6-max-preview"
    agent_strong_model: str = "qwen3.7-plus"
    agent_llm_model: str = "qwen3.6-plus"
    agent_intent_model: str = "qwen3.6-max-preview"
    agent_safety_model: str = "qwen3.6-max-preview"
    agent_planner_model: str = "qwen3.6-max-preview"
    agent_rag_model: str = "qwen3.6-max-preview"
    agent_research_model: str = "qwen3.7-plus"
    agent_artifact_model: str = "qwen3.6-plus"
    agent_memory_model: str = "qwen3.6-max-preview"
    agent_skill_model: str = "qwen3.6-max-preview"
    agent_final_model: str = "qwen3.6-plus"
    agent_embedding_provider: str = "aliyun"
    agent_embedding_model: str = "text-embedding-v4"
    agent_intent_llm_enabled: bool = True
    agent_planner_llm_enabled: bool = False
    agent_memory_llm_enabled: bool = True
    agent_skill_llm_enabled: bool = True
    agent_safety_llm_enabled: bool = True
    agent_llm_usage_log_enabled: bool = True
    agent_llm_log_prompt_preview: bool = False
    agent_llm_log_raw_output: bool = False
    # Max recent chat messages injected into the LLM context per turn.
    # Only the last N messages are loaded; earlier messages rely on
    # conversation_summary (running) + recalled historical segments.
    conversation_recent_message_limit: int = 24
    # ── Running conversation summary (incremental, after each assistant turn) ──
    enable_conversation_summary: bool = True
    # ── Conversation segment: freeze old messages into indexed summary blocks ──
    # Controls whether the system creates compressed historical segments from
    # long conversations. These segments are later recalled via vector search
    # to keep early-turn context alive even after hundreds of messages.
    enable_conversation_segment_creation: bool = True
    # Controls whether the context builder recalls relevant historical segments
    # via Qdrant vector search (fallback to PostgreSQL ILIKE).
    enable_conversation_segment_recall: bool = True
    # Number of messages per frozen historical segment.
    conversation_summary_segment_size: int = 24
    # Max number of recalled segments injected into the LLM context per turn.
    conversation_segment_recall_limit: int = 5
    # Minimum similarity / keyword-match score for a segment to be recalled.
    conversation_segment_min_score: float = 0.15
    # Token budget ceiling for all recalled segment text combined.
    conversation_segment_max_tokens: int = 1800
    # Qdrant collection name for conversation segment vectors.
    conversation_segment_vector_collection: str = "conversation_summary_segments"

    feed_real_sources_enabled: bool = True
    feed_allow_mock_data: bool = False
    feed_home_card_count: int = 3
    feed_require_real_cards: bool = True
    feed_source_arxiv_enabled: bool = True
    feed_source_github_enabled: bool = True
    feed_source_duckduckgo_enabled: bool = True
    feed_source_tavily_enabled: bool = True
    feed_source_serpapi_enabled: bool = True
    feed_source_manual_seed_enabled: bool = True
    feed_refresh_on_home_empty: bool = True
    feed_refresh_min_real_cards: int = 3
    feed_refresh_timeout_seconds: int = 45
    feed_refresh_dedup_enabled: bool = True
    feed_refresh_total_limit: int = 5
    feed_refresh_explicit_min: int = 2
    feed_refresh_adjacent_min: int = 2
    feed_refresh_far_min: int = 1
    agent_timeline_enabled: bool = True
    agent_langgraph_status_enabled: bool = True
    agent_langgraph_status_max_steps: int = 12
    agent_chat_real_messages_enabled: bool = True
    agent_supervisor_enabled: bool = True
    agent_supervisor_shadow_policy_enabled: bool = True
    agent_supervisor_shadow_metrics_enabled: bool = True
    agent_supervisor_control_enabled: bool = False
    agent_replanner_control_enabled: bool = False
    agent_llm_supervisor_enabled: bool = True
    agent_llm_supervisor_mode: str = "full"
    agent_llm_supervisor_model: str = "qwen-plus"
    agent_llm_supervisor_temperature: float = 0
    agent_llm_supervisor_timeout_seconds: int = 20
    agent_langgraph_checkpointer_enabled: bool = True
    # Approval pause always uses LangGraph interrupt() / Command(resume=...)
    # Pending approvals older than this are auto-expired (cannot be approved/rejected).
    # Expired approvals' checkpoints enter cleanup rotation after the expired TTL.
    agent_approval_pending_ttl_hours: int = 24
    # Background checkpoint cleanup interval in minutes.  0 = disabled.
    # Cleans completed/cancelled/expired checkpoints (TTL: 7d) and failed (30d),
    # plus any orphan checkpoints whose agent_runs row has been deleted.
    agent_checkpoint_cleanup_interval_minutes: int = 60
    agent_checkpoint_cleanup_enabled: bool = True
    # Checkpointer backend: "postgres" (production) | "redis" (experimental) | "memory" (dev/test only)
    agent_checkpointer_backend: str = "postgres"
    # True → fail fast at startup if durable checkpoint storage is unavailable.
    # Production must be True.  Only set False for local dev with no PostgreSQL.
    agent_checkpointer_require_durable: bool = True
    agent_checkpointer_database_url: str = ""  # if empty, uses main database_url
    # Redis checkpointer configuration (when backend=redis)
    redis_password: str = ""
    redis_checkpointer_key_prefix: str = "langgraph:checkpoint:"
    # Qwen model tier configuration
    qwen_fast_model: str = "qwen3.6-max-preview"
    qwen_balanced_model: str = "qwen3.6-plus"
    qwen_advanced_model: str = "qwen3.6-max-preview"
    qwen_vision_model: str = "qwen3.6-plus"
    max_chat_upload_bytes: int = 20 * 1024 * 1024
    # ── Local File Tools ───────────────────────────────────────────
    local_tools_enabled: bool = True
    local_tools_workspace_dir: str = "./agent_workspace"
    local_tools_allow_delete: bool = False
    local_tools_max_read_chars: int = 8000
    local_tools_max_write_chars: int = 20000
    # ── Email Provider ────────────────────────────────────────────
    email_provider: str = "mock"  # mock | smtp
    email_from: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def database_url(self) -> str:
        password = self.postgres_password
        encoded_password = password  # consider url-encoding if password has special chars
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
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
