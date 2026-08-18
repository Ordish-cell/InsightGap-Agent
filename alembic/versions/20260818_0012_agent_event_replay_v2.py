"""add durable Agent event envelope fields

Revision ID: 20260818_0012
Revises: 20260622_0011
Create Date: 2026-08-18
"""

from alembic import op
import sqlalchemy as sa


revision = "20260818_0012"
down_revision = "20260622_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_events", sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("agent_events", sa.Column("visibility", sa.String(length=32), nullable=False, server_default="trace"))
    op.add_column("agent_events", sa.Column("display_channel", sa.String(length=32), nullable=False, server_default="status"))

    op.execute("UPDATE agent_events SET visibility = 'user', display_channel = 'thinking' WHERE event_type IN ('visible_thought_delta', 'visible_progress_delta', 'milestone_started', 'milestone_completed')")
    op.execute("UPDATE agent_events SET visibility = 'user', display_channel = 'answer' WHERE event_type IN ('answer_started', 'answer_delta', 'answer_completed')")
    op.execute("UPDATE agent_events SET visibility = 'user', display_channel = 'tool' WHERE event_type IN ('tool_call_started', 'tool_call_delta', 'tool_call_completed', 'tool_call_failed')")
    op.execute("UPDATE agent_events SET visibility = 'user', display_channel = 'status' WHERE event_type IN ('approval_required', 'approval_granted', 'approval_rejected', 'run_created', 'run_completed', 'run_failed', 'run_paused', 'run_resumed', 'run_interrupted')")
    op.create_index("ix_agent_events_user_run_id", "agent_events", ["user_id", "run_id", "id"])


def downgrade() -> None:
    op.drop_index("ix_agent_events_user_run_id", table_name="agent_events")
    op.drop_column("agent_events", "display_channel")
    op.drop_column("agent_events", "visibility")
    op.drop_column("agent_events", "schema_version")
