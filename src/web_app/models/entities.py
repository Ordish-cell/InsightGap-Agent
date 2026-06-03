from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def now() -> datetime:
    return datetime.utcnow()


@dataclass
class User:
    id: int | None = None
    email: str = ""
    hashed_password: str = ""
    nickname: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class UserProfile:
    id: int | None = None
    user_id: int | None = None
    segment: str = "general_user"
    goals: list[Any] = field(default_factory=list)
    explicit_interests: list[Any] = field(default_factory=list)
    adjacent_domains: list[Any] = field(default_factory=list)
    far_domains: list[Any] = field(default_factory=list)
    disliked_topics: list[Any] = field(default_factory=list)
    preferred_outputs: list[Any] = field(default_factory=list)
    risk_preference: str = "normal"
    feed_ratio_config: dict[str, float] = field(default_factory=lambda: {"explicit_related": 0.30, "adjacent_domain": 0.40, "far_domain": 0.30})
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class InfoSource:
    id: int | None = None
    name: str = ""
    source_type: str = "web"
    base_url: str = ""
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class InfoItem:
    id: int | None = None
    source_id: int | None = None
    title: str = ""
    summary: str = ""
    content: str = ""
    source_url: str = ""
    source_type: str = "web"
    author: str = ""
    published_at: datetime | None = None
    language: str = "zh"
    entities: list[Any] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class FeedCard:
    id: int | None = None
    user_id: int | None = None
    info_item_id: int | None = None
    card_type: str = "insight"
    title: str = ""
    one_sentence_value: str = ""
    why_you: str = ""
    information_gap: str = ""
    evidence: list[Any] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    score_detail: dict[str, Any] = field(default_factory=dict)
    final_score: float = 0.0
    exposure_bucket: str = "explicit_related"
    status: str = "new"
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class Document:
    id: int | None = None
    user_id: int | None = None
    filename: str = ""
    file_path: str = ""
    file_type: str = ""
    source_type: str = "user_upload"
    status: str = "created"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class DocumentChunk:
    id: int | None = None
    document_id: int | None = None
    user_id: int | None = None
    chunk_index: int = 0
    content: str = ""
    token_count: int = 0
    qdrant_point_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)


@dataclass
class Memory:
    id: int | None = None
    user_id: int | None = None
    memory_type: str = "working"
    content: str = ""
    importance: float = 0.0
    source_type: str = ""
    source_id: str = ""
    qdrant_point_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class Skill:
    id: int | None = None
    user_id: int | None = None
    name: str = ""
    description: str = ""
    trigger_text: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    context_recipe: list[Any] = field(default_factory=list)
    tool_plan: list[Any] = field(default_factory=list)
    output_schema: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "read_only"
    eval_checks: list[Any] = field(default_factory=list)
    status: str = "draft"
    version: int = 1
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class AgentRun:
    id: int | None = None
    user_id: int | None = None
    run_type: str = "chat"
    mode: str = "react"
    status: str = "created"
    user_input: str = ""
    graph_state: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    error_message: str = ""
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class AgentStep:
    id: int | None = None
    run_id: int | None = None
    node_name: str = ""
    agent_name: str = ""
    action_type: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    started_at: datetime | None = None
    ended_at: datetime | None = None


@dataclass
class MCPServer:
    id: int | None = None
    name: str = ""
    description: str = ""
    server_url: str = ""
    enabled: bool = True
    auth_config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class MCPTool:
    id: int | None = None
    server_id: int | None = None
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    permission_level: str = "L0_READ_ONLY"
    approval_required: bool = False
    enabled: bool = True
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class ToolCall:
    id: int | None = None
    run_id: int | None = None
    user_id: int | None = None
    tool_name: str = ""
    mcp_tool_id: int | None = None
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    permission_level: str = "L0_READ_ONLY"
    status: str = "created"
    error_message: str = ""
    created_at: datetime = field(default_factory=now)


@dataclass
class Approval:
    id: int | None = None
    user_id: int | None = None
    run_id: int | None = None
    approval_type: str = "external_write"
    title: str = ""
    description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)


@dataclass
class Artifact:
    id: int | None = None
    user_id: int | None = None
    run_id: int | None = None
    artifact_type: str = "markdown"
    title: str = ""
    file_path: str = ""
    public_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)


@dataclass
class EvalRecord:
    id: int | None = None
    user_id: int | None = None
    run_id: int | None = None
    target_type: str = "answer"
    target_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)
