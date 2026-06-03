"""add feed feedback

Revision ID: 20260602_0002
Revises: 20260602_0001
Create Date: 2026-06-02
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260602_0002"
down_revision: str | Sequence[str] | None = "20260602_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feed_feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("card_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["card_id"], ["feed_cards.id"]),
    )
    op.create_index("ix_feed_feedback_user_id", "feed_feedback", ["user_id"])
    op.create_index("ix_feed_feedback_card_id", "feed_feedback", ["card_id"])
    op.create_index("ix_feed_feedback_user_card", "feed_feedback", ["user_id", "card_id"])


def downgrade() -> None:
    op.drop_index("ix_feed_feedback_user_card", table_name="feed_feedback")
    op.drop_index("ix_feed_feedback_card_id", table_name="feed_feedback")
    op.drop_index("ix_feed_feedback_user_id", table_name="feed_feedback")
    op.drop_table("feed_feedback")
