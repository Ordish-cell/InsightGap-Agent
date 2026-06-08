from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.web_app.db.base import Base


def json_default() -> list[Any]:
    return []


def feed_ratio_default() -> dict[str, float]:
    return {"explicit_related": 0.30, "adjacent_domain": 0.40, "far_domain": 0.30}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)

    profile: Mapped["UserProfile"] = relationship(back_populates="user", uselist=False)


class UserProfile(Base, TimestampMixin):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True, nullable=False)
    segment: Mapped[str] = mapped_column(String(32), default="general_user", nullable=False)
    goals: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    explicit_interests: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    adjacent_domains: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    far_domains: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    disliked_topics: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    preferred_outputs: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    risk_preference: Mapped[str] = mapped_column(String(32), default="normal", nullable=False)
    feed_ratio_config: Mapped[dict[str, float]] = mapped_column(JSON, default=feed_ratio_default, nullable=False)
    last_feed_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_feed_refresh_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="profile")

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("feed_ratio_config", feed_ratio_default())
        kwargs.setdefault("goals", [])
        kwargs.setdefault("explicit_interests", [])
        kwargs.setdefault("adjacent_domains", [])
        kwargs.setdefault("far_domains", [])
        kwargs.setdefault("disliked_topics", [])
        kwargs.setdefault("preferred_outputs", [])
        super().__init__(**kwargs)


class InfoSource(Base, TimestampMixin):
    __tablename__ = "info_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    base_url: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class InfoItem(Base, TimestampMixin):
    __tablename__ = "info_items"
    __table_args__ = (Index("ix_info_items_content_hash", "content_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("info_sources.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), default="web", nullable=False)
    author: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    language: Mapped[str] = mapped_column(String(16), default="zh", nullable=False)
    entities: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    topics: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), default="", nullable=False)


class FeedCard(Base, TimestampMixin):
    __tablename__ = "feed_cards"
    __table_args__ = (Index("ix_feed_cards_user_status", "user_id", "status"), Index("ix_feed_cards_batch_id", "batch_id"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    info_item_id: Mapped[int] = mapped_column(ForeignKey("info_items.id"), nullable=False)
    card_type: Mapped[str] = mapped_column(String(32), default="insight", nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    one_sentence_value: Mapped[str] = mapped_column(Text, default="", nullable=False)
    why_you: Mapped[str] = mapped_column(Text, default="", nullable=False)
    information_gap: Mapped[str] = mapped_column(Text, default="", nullable=False)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    suggested_actions: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    score_detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    exposure_bucket: Mapped[str] = mapped_column(String(32), default="explicit_related", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="new", nullable=False)
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class FeedFeedback(Base):
    __tablename__ = "feed_feedback"
    __table_args__ = (Index("ix_feed_feedback_user_card", "user_id", "card_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    card_id: Mapped[int] = mapped_column(ForeignKey("feed_cards.id"), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_type: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), default="user_upload", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Memory(Base, TimestampMixin):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(32), default="working", index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    source_id: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Skill(Base, TimestampMixin):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    trigger_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    context_recipe: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    tool_plan: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    output_schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    safety_level: Mapped[str] = mapped_column(String(32), default="read_only", nullable=False)
    eval_checks: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class AgentRun(Base, TimestampMixin):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(64), default="", index=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), default="", index=True, nullable=False)
    run_type: Mapped[str] = mapped_column(String(64), default="chat", nullable=False)
    mode: Mapped[str] = mapped_column(String(32), default="react", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    user_input: Mapped[str] = mapped_column(Text, default="", nullable=False)
    graph_state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    result_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    final_answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    final_response: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    langgraphstatus_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentConversation(Base, TimestampMixin):
    __tablename__ = "agent_conversations"
    __table_args__ = (
        Index("ix_agent_conversations_user_status", "user_id", "status"),
        Index("ix_agent_conversations_user_conversation", "user_id", "conversation_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="agent_page", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), default="", index=True, nullable=False)
    selected_feed_card_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_feed_card_title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    last_message_preview: Mapped[str] = mapped_column(Text, default="", nullable=False)
    last_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_active_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class AgentChatMessage(Base, TimestampMixin):
    __tablename__ = "agent_chat_messages"
    __table_args__ = (
        Index("ix_agent_chat_messages_conversation_created", "conversation_id", "created_at"),
        Index("ix_agent_chat_messages_user_conversation", "user_id", "conversation_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    thread_id: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    langgraphstatus_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    steps_json: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class AgentStep(Base):
    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    action_type: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    input: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentEvent(Base):
    __tablename__ = "agent_events"
    __table_args__ = (Index("ix_agent_events_run_id", "run_id"), Index("ix_agent_events_user_run", "user_id", "run_id"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), default="", index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class LLMCall(Base):
    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_run_id", "run_id"), Index("ix_llm_calls_user_run", "user_id", "run_id"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    thread_id: Mapped[str] = mapped_column(String(255), default="", index=True, nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    node_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    tier: Mapped[str] = mapped_column(String(32), nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_input_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_output_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class MCPServer(Base, TimestampMixin):
    __tablename__ = "mcp_servers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    server_url: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    auth_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class MCPTool(Base, TimestampMixin):
    __tablename__ = "mcp_tools"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    server_id: Mapped[int | None] = mapped_column(ForeignKey("mcp_servers.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output_schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    permission_level: Mapped[str] = mapped_column(String(32), default="L0_READ_ONLY", nullable=False)
    approval_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mcp_tool_id: Mapped[int | None] = mapped_column(ForeignKey("mcp_tools.id"), nullable=True)
    input: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    permission_level: Mapped[str] = mapped_column(String(32), default="L0_READ_ONLY", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Approval(Base, TimestampMixin):
    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    approval_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    public_url: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class EvalRecord(Base):
    __tablename__ = "eval_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    feedback: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class ResearchRun(Base, TimestampMixin):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    feed_card_id: Mapped[int | None] = mapped_column(ForeignKey("feed_cards.id"), index=True, nullable=True)
    agent_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"), index=True, nullable=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    findings: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    risks: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    opportunities: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    suggested_actions: Mapped[list[Any]] = mapped_column(JSON, default=json_default, nullable=False)
    markdown_report: Mapped[str] = mapped_column(Text, default="", nullable=False)
    artifact_id: Mapped[int | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    skill_draft_id: Mapped[int | None] = mapped_column(ForeignKey("skills.id"), nullable=True)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
