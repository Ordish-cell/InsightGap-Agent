"""Scoped memories, explicit consent and durable maintenance. No model backfill."""
from alembic import op
import sqlalchemy as sa
import re

revision = "20260930_0016"
down_revision = "20260909_0015"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("user_profiles", sa.Column("use_memory", sa.Boolean(), server_default=sa.true(), nullable=False))
    op.add_column("user_profiles", sa.Column("generate_memory", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.add_column("user_profiles", sa.Column("memory_settings_version", sa.Integer(), server_default="0", nullable=False))
    op.add_column("agent_runs", sa.Column("memory_policy", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("memories", sa.Column("scope", sa.String(32), server_default="legacy_unscoped", nullable=False))
    op.add_column("memories", sa.Column("scope_id", sa.String(64), server_default="", nullable=False))
    op.create_index("ix_memories_scope", "memories", ["scope"])
    op.create_index("ix_memories_scope_id", "memories", ["scope_id"])
    op.create_table("memory_maintenance_tasks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_message_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("policy", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("run_id", name="uq_memory_maintenance_run"))
    op.create_index("ix_memory_maintenance_tasks_user_id", "memory_maintenance_tasks", ["user_id"])
    op.create_index("ix_memory_maintenance_tasks_conversation_id", "memory_maintenance_tasks", ["conversation_id"])
    bind = op.get_bind()
    memories = sa.table("memories", sa.column("id"), sa.column("user_id"), sa.column("metadata_json", sa.JSON()),
                        sa.column("scope"), sa.column("scope_id"), sa.column("source_type"), sa.column("source_id"))
    conversations = sa.table("agent_conversations", sa.column("conversation_id"), sa.column("user_id"))
    runs = sa.table("agent_runs", sa.column("id"), sa.column("user_id"), sa.column("conversation_id"))
    PERSONAL_CATEGORIES = {"preferred_name", "response_language", "script_preference", "name_preference", "language_preference", "tone_preference", "answer_preference", "output_preference"}
    owners = {(r.user_id, r.conversation_id) for r in bind.execute(sa.select(conversations))}
    run_owners = {(r.user_id, str(r.id)): r.conversation_id for r in bind.execute(sa.select(runs))}
    for row in bind.execute(sa.select(memories)).mappings().all():
        meta = row["metadata_json"] or {}
        cid = meta.get("conversation_id")
        if not cid or (row["user_id"], cid) not in owners:
            cid = run_owners.get((row["user_id"], str(meta.get("run_id") or meta.get("agent_run_id") or meta.get("source_run_id"))))
        if not cid and row["source_type"] in {"conversation", "agent_conversation"}:
            cid = row["source_id"]
        if not cid and row["source_type"] in {"run", "agent_run"}:
            cid = run_owners.get((row["user_id"], row["source_id"]))
        scope, scope_id = "legacy_unscoped", ""
        contextual = bool(re.search(r"本项目|这个项目|当前项目|本次|这次|本轮|暂时|this (?:project|task|time)|for now", str(meta.get("source_quote") or ""), re.I))
        if meta.get("category") in PERSONAL_CATEGORIES and not contextual:
            scope = "user"
        elif cid and (row["user_id"], cid) in owners:
            scope, scope_id = "conversation", cid
        fixed = {"preferred_name": "preferred_name", "name_preference": "preferred_name",
                 "response_language": "response_language", "language_preference": "response_language",
                 "script_preference": "script_preference"}.get(meta.get("category"))
        if fixed:
            meta = {**meta, "fact_key": fixed}
        bind.execute(memories.update().where(memories.c.id == row["id"]).values(scope=scope, scope_id=scope_id, metadata_json=meta))


def downgrade():
    op.drop_table("memory_maintenance_tasks")
    op.drop_index("ix_memories_scope_id", table_name="memories")
    op.drop_index("ix_memories_scope", table_name="memories")
    op.drop_column("memories", "scope_id")
    op.drop_column("memories", "scope")
    op.drop_column("agent_runs", "memory_policy")
    for name in ("memory_settings_version", "generate_memory", "use_memory"):
        op.drop_column("user_profiles", name)
