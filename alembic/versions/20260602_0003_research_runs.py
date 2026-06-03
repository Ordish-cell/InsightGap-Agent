"""add research runs

Revision ID: 20260602_0003
Revises: 20260602_0002
Create Date: 2026-06-02
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260602_0003"
down_revision: str | Sequence[str] | None = "20260602_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("feed_card_id", sa.Integer(), nullable=True),
        sa.Column("agent_run_id", sa.Integer(), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("findings", sa.JSON(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("risks", sa.JSON(), nullable=True),
        sa.Column("opportunities", sa.JSON(), nullable=True),
        sa.Column("suggested_actions", sa.JSON(), nullable=True),
        sa.Column("markdown_report", sa.Text(), nullable=False),
        sa.Column("artifact_id", sa.Integer(), nullable=True),
        sa.Column("skill_draft_id", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["feed_card_id"], ["feed_cards.id"]),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"]),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["skill_draft_id"], ["skills.id"]),
    )
    op.create_index("ix_research_runs_user_id", "research_runs", ["user_id"])
    op.create_index("ix_research_runs_feed_card_id", "research_runs", ["feed_card_id"])
    op.create_index("ix_research_runs_agent_run_id", "research_runs", ["agent_run_id"])


def downgrade() -> None:
    op.drop_index("ix_research_runs_agent_run_id", table_name="research_runs")
    op.drop_index("ix_research_runs_feed_card_id", table_name="research_runs")
    op.drop_index("ix_research_runs_user_id", table_name="research_runs")
    op.drop_table("research_runs")
