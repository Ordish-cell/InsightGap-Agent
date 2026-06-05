"""add agent conversations

Revision ID: 20260605_0006
Revises: 20260604_0005
Create Date: 2026-06-05
"""

from alembic import op
import sqlalchemy as sa


revision = "20260605_0006"
down_revision = "20260604_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("conversation_id", sa.String(length=64), nullable=False, server_default=""))
    op.add_column("agent_runs", sa.Column("thread_id", sa.String(length=255), nullable=False, server_default=""))
    op.add_column("agent_runs", sa.Column("final_answer", sa.Text(), nullable=False, server_default=""))
    op.add_column("agent_runs", sa.Column("final_response", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("agent_runs", sa.Column("langgraphstatus_json", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("agent_runs", sa.Column("elapsed_ms", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("agent_runs", sa.Column("completed_at", sa.DateTime(), nullable=True))
    op.create_index("ix_agent_runs_conversation_id", "agent_runs", ["conversation_id"])
    op.create_index("ix_agent_runs_thread_id", "agent_runs", ["thread_id"])

    op.create_table(
        "agent_conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="agent_page"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("thread_id", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("selected_feed_card_id", sa.Integer(), nullable=True),
        sa.Column("selected_feed_card_title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("last_message_preview", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_run_id", sa.Integer(), sa.ForeignKey("agent_runs.id"), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("last_active_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_agent_conversations_conversation_id", "agent_conversations", ["conversation_id"], unique=True)
    op.create_index("ix_agent_conversations_thread_id", "agent_conversations", ["thread_id"])
    op.create_index("ix_agent_conversations_user_id", "agent_conversations", ["user_id"])
    op.create_index("ix_agent_conversations_user_status", "agent_conversations", ["user_id", "status"])
    op.create_index("ix_agent_conversations_user_conversation", "agent_conversations", ["user_id", "conversation_id"])

    op.create_table(
        "agent_chat_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("message_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("agent_runs.id"), nullable=True),
        sa.Column("thread_id", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("langgraphstatus_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("steps_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("error_message", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_agent_chat_messages_message_id", "agent_chat_messages", ["message_id"], unique=True)
    op.create_index("ix_agent_chat_messages_conversation_id", "agent_chat_messages", ["conversation_id"])
    op.create_index("ix_agent_chat_messages_user_id", "agent_chat_messages", ["user_id"])
    op.create_index("ix_agent_chat_messages_conversation_created", "agent_chat_messages", ["conversation_id", "created_at"])
    op.create_index("ix_agent_chat_messages_user_conversation", "agent_chat_messages", ["user_id", "conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_chat_messages_user_conversation", table_name="agent_chat_messages")
    op.drop_index("ix_agent_chat_messages_conversation_created", table_name="agent_chat_messages")
    op.drop_index("ix_agent_chat_messages_user_id", table_name="agent_chat_messages")
    op.drop_index("ix_agent_chat_messages_conversation_id", table_name="agent_chat_messages")
    op.drop_index("ix_agent_chat_messages_message_id", table_name="agent_chat_messages")
    op.drop_table("agent_chat_messages")

    op.drop_index("ix_agent_conversations_user_conversation", table_name="agent_conversations")
    op.drop_index("ix_agent_conversations_user_status", table_name="agent_conversations")
    op.drop_index("ix_agent_conversations_user_id", table_name="agent_conversations")
    op.drop_index("ix_agent_conversations_thread_id", table_name="agent_conversations")
    op.drop_index("ix_agent_conversations_conversation_id", table_name="agent_conversations")
    op.drop_table("agent_conversations")

    op.drop_index("ix_agent_runs_thread_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_conversation_id", table_name="agent_runs")
    op.drop_column("agent_runs", "completed_at")
    op.drop_column("agent_runs", "elapsed_ms")
    op.drop_column("agent_runs", "langgraphstatus_json")
    op.drop_column("agent_runs", "final_response")
    op.drop_column("agent_runs", "final_answer")
    op.drop_column("agent_runs", "thread_id")
    op.drop_column("agent_runs", "conversation_id")
