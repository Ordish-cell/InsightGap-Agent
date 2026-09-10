"""Add ordinary-chat controls; apply explicitly with the application stopped."""
from alembic import op
import sqlalchemy as sa

revision = "20260907_0014"
down_revision = "20260818_0013"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("agent_runs", sa.Column("supersedes_run_id", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_agent_run_supersedes", "agent_runs", "agent_runs", ["supersedes_run_id"], ["id"], ondelete="SET NULL")
    op.add_column("agent_runs", sa.Column("chat_control_phase", sa.String(32), server_default="disabled", nullable=False))
    op.create_table(
        "agent_run_controls",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("successor_run_id", sa.Integer(), sa.ForeignKey("agent_runs.id", ondelete="SET NULL")),
        sa.Column("client_command_id", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_agent_control_request", "agent_run_controls", ["user_id", "client_command_id"], unique=True)
    op.create_index("ix_agent_run_controls_run_id", "agent_run_controls", ["run_id"])


def downgrade():
    op.drop_table("agent_run_controls")
    op.drop_constraint("fk_agent_run_supersedes", "agent_runs", type_="foreignkey")
    op.drop_column("agent_runs", "supersedes_run_id")
    op.drop_column("agent_runs", "chat_control_phase")
