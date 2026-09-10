"""Durable conversation deletion jobs. Apply explicitly, never on startup."""
from alembic import op
import sqlalchemy as sa

revision = "20260909_0015"
down_revision = "20260907_0014"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("conversation_deletion_tasks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("phase", sa.String(32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("progress", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "conversation_id", name="uq_conversation_deletion_owner"))
    op.create_index("ix_conversation_deletion_tasks_user_id", "conversation_deletion_tasks", ["user_id"])


def downgrade():
    op.drop_table("conversation_deletion_tasks")
