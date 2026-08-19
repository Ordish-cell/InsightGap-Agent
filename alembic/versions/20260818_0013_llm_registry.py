"""add user-scoped LLM registry

Revision ID: 20260818_0013
Revises: 20260818_0012
Create Date: 2026-08-18
"""

from alembic import op
import sqlalchemy as sa


revision = "20260818_0013"
down_revision = "20260818_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if "agent_runs" in sa.inspect(bind).get_table_names():
        active_runs = bind.execute(
            sa.text("SELECT COUNT(*) FROM agent_runs WHERE status IN ('created', 'running', 'resuming', 'waiting_approval')")
        ).scalar_one()
        if active_runs:
            raise RuntimeError(
                f"LLM registry cutover blocked: {active_runs} active Agent Run(s) must finish or be cancelled first"
            )
    op.create_table(
        "llm_connections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("protocol", sa.String(64), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("encrypted_secrets", sa.Text(), nullable=False, server_default=""),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(32), nullable=False, server_default="draft"),
        sa.Column("last_test_status", sa.String(32), nullable=False, server_default="untested"),
        sa.Column("last_test_error", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_tested_at", sa.DateTime(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_llm_connections_user_id", "llm_connections", ["user_id"])
    op.create_index("ix_llm_connections_user_status", "llm_connections", ["user_id", "status"])
    op.create_table(
        "llm_models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("connection_id", sa.Integer(), sa.ForeignKey("llm_connections.id"), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("source", sa.String(32), nullable=False, server_default="manual"),
        sa.Column("capabilities_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_llm_models_connection_id", "llm_models", ["connection_id"])
    op.create_index("ix_llm_models_connection_enabled", "llm_models", ["connection_id", "enabled"])
    op.create_index("ux_llm_models_connection_model", "llm_models", ["connection_id", "model_id"], unique=True)
    op.add_column("user_profiles", sa.Column("default_llm_model_id", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_user_profiles_default_llm_model", "user_profiles", "llm_models", ["default_llm_model_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    op.drop_constraint("fk_user_profiles_default_llm_model", "user_profiles", type_="foreignkey")
    op.drop_column("user_profiles", "default_llm_model_id")
    op.drop_table("llm_models")
    op.drop_table("llm_connections")
