"""add agent events

Revision ID: 20260604_0004
Revises: 20260602_0003
Create Date: 2026-06-04
"""

from alembic import op
import sqlalchemy as sa


revision = "20260604_0004"
down_revision = "20260602_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("node_name", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_agent_events_run_id", "agent_events", ["run_id"])
    op.create_index("ix_agent_events_thread_id", "agent_events", ["thread_id"])
    op.create_index("ix_agent_events_user_id", "agent_events", ["user_id"])
    op.create_index("ix_agent_events_user_run", "agent_events", ["user_id", "run_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_events_user_run", table_name="agent_events")
    op.drop_index("ix_agent_events_user_id", table_name="agent_events")
    op.drop_index("ix_agent_events_thread_id", table_name="agent_events")
    op.drop_index("ix_agent_events_run_id", table_name="agent_events")
    op.drop_table("agent_events")
