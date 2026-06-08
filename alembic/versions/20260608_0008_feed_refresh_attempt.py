"""add last_feed_refresh_attempt_at to prevent infinite refresh loops

Revision ID: 20260608_0008
Revises: 20260608_0007
Create Date: 2026-06-08
"""

from alembic import op
import sqlalchemy as sa


revision = "20260608_0008"
down_revision = "20260608_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("user_profiles", sa.Column("last_feed_refresh_attempt_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("user_profiles", "last_feed_refresh_attempt_at")
