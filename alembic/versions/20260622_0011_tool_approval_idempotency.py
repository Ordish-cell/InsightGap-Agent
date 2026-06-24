"""Add idempotency keys for tool approvals.

Revision ID: 20260622_0011
Revises: 20260619_0010
Create Date: 2026-06-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260622_0011"
down_revision = "20260619_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tool_calls", sa.Column("idempotency_key", sa.String(length=255), nullable=True))
    op.add_column("approvals", sa.Column("idempotency_key", sa.String(length=255), nullable=True))
    op.create_index("ix_tool_calls_idempotency_key", "tool_calls", ["idempotency_key"], unique=True)
    op.create_index("ix_approvals_idempotency_key", "approvals", ["idempotency_key"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_approvals_idempotency_key", table_name="approvals")
    op.drop_index("ix_tool_calls_idempotency_key", table_name="tool_calls")
    op.drop_column("approvals", "idempotency_key")
    op.drop_column("tool_calls", "idempotency_key")
