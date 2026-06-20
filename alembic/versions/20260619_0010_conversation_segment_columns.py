"""add start_message_created_at, end_message_created_at, message_count to summary segments
and replace non-unique index with unique constraint on msg range.

Revision ID: 20260619_0010
Revises: 20260619_0009
Create Date: 2026-06-19
"""

from alembic import op
import sqlalchemy as sa


revision = "20260619_0010"
down_revision = "20260619_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns
    op.add_column(
        "agent_conversation_summary_segments",
        sa.Column("start_message_created_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "agent_conversation_summary_segments",
        sa.Column("end_message_created_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "agent_conversation_summary_segments",
        sa.Column("message_count", sa.Integer(), server_default="0", nullable=False),
    )

    # Replace non-unique index with unique constraint to prevent duplicate segments
    op.drop_index(
        "ix_agent_conv_segments_msg_range",
        table_name="agent_conversation_summary_segments",
    )
    op.create_index(
        "ix_agent_conv_segments_msg_range",
        "agent_conversation_summary_segments",
        ["conversation_id", "start_message_id", "end_message_id"],
        unique=True,
    )


def downgrade() -> None:
    # Revert to non-unique index
    op.drop_index(
        "ix_agent_conv_segments_msg_range",
        table_name="agent_conversation_summary_segments",
    )
    op.create_index(
        "ix_agent_conv_segments_msg_range",
        "agent_conversation_summary_segments",
        ["conversation_id", "start_message_id", "end_message_id"],
        unique=False,
    )

    op.drop_column("agent_conversation_summary_segments", "message_count")
    op.drop_column("agent_conversation_summary_segments", "end_message_created_at")
    op.drop_column("agent_conversation_summary_segments", "start_message_created_at")
