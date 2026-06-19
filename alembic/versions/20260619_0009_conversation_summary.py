"""add agent_conversation_summaries and agent_conversation_summary_segments

Revision ID: 20260619_0009
Revises: 20260608_0008
Create Date: 2026-06-19
"""

from alembic import op
import sqlalchemy as sa


revision = "20260619_0009"
down_revision = "20260608_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_conversation_summaries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.String(64), unique=True, index=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), index=True, nullable=False),
        sa.Column("summary_text", sa.Text(), default="", nullable=False),
        sa.Column("facts_json", sa.JSON(), default=list, nullable=False),
        sa.Column("decisions_json", sa.JSON(), default=list, nullable=False),
        sa.Column("open_threads_json", sa.JSON(), default=list, nullable=False),
        sa.Column("preferences_json", sa.JSON(), default=list, nullable=False),
        sa.Column("entities_json", sa.JSON(), default=list, nullable=False),
        sa.Column("last_message_id", sa.Integer(), sa.ForeignKey("agent_chat_messages.id"), nullable=True),
        sa.Column("covered_message_count", sa.Integer(), default=0, nullable=False),
        sa.Column("summary_version", sa.Integer(), default=1, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_agent_conv_summaries_user_conv",
        "agent_conversation_summaries",
        ["user_id", "conversation_id"],
    )
    op.create_index(
        "ix_agent_conv_summaries_updated_at",
        "agent_conversation_summaries",
        ["updated_at"],
    )

    op.create_table(
        "agent_conversation_summary_segments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.String(64), index=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), index=True, nullable=False),
        sa.Column("start_message_id", sa.Integer(), sa.ForeignKey("agent_chat_messages.id"), nullable=True),
        sa.Column("end_message_id", sa.Integer(), sa.ForeignKey("agent_chat_messages.id"), nullable=True),
        sa.Column("summary_text", sa.Text(), default="", nullable=False),
        sa.Column("keywords_json", sa.JSON(), default=list, nullable=False),
        sa.Column("facts_json", sa.JSON(), default=list, nullable=False),
        sa.Column("embedding_id", sa.String(64), default="", nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_agent_conv_segments_user_conv",
        "agent_conversation_summary_segments",
        ["user_id", "conversation_id"],
    )
    op.create_index(
        "ix_agent_conv_segments_msg_range",
        "agent_conversation_summary_segments",
        ["conversation_id", "start_message_id", "end_message_id"],
    )


def downgrade() -> None:
    op.drop_table("agent_conversation_summary_segments")
    op.drop_table("agent_conversation_summaries")
