"""add feed batch tracking and daily refresh

Revision ID: 20260608_0007
Revises: 20260605_0006
Create Date: 2026-06-08
"""

from alembic import op
import sqlalchemy as sa


revision = "20260608_0007"
down_revision = "20260605_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("feed_cards", sa.Column("batch_id", sa.String(length=64), nullable=True))
    op.add_column("feed_cards", sa.Column("generated_at", sa.DateTime(), nullable=True))
    op.create_index("ix_feed_cards_batch_id", "feed_cards", ["batch_id"])
    op.add_column("user_profiles", sa.Column("last_feed_refreshed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_index("ix_feed_cards_batch_id", table_name="feed_cards")
    op.drop_column("feed_cards", "generated_at")
    op.drop_column("feed_cards", "batch_id")
    op.drop_column("user_profiles", "last_feed_refreshed_at")
