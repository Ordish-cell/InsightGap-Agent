"""Migration verifies deterministic legacy classification and reversible DDL."""
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


def test_memory_migration_and_downgrade():
    spec = importlib.util.spec_from_file_location("memory_migration", Path(__file__).resolve().parents[3] / "alembic/versions/20260930_0016_memory_mechanism.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    users = sa.Table("users", metadata, sa.Column("id", sa.Integer(), primary_key=True))
    profiles = sa.Table("user_profiles", metadata, sa.Column("id", sa.Integer(), primary_key=True))
    runs = sa.Table("agent_runs", metadata, sa.Column("id", sa.Integer(), primary_key=True), sa.Column("user_id", sa.Integer()), sa.Column("conversation_id", sa.String()))
    convs = sa.Table("agent_conversations", metadata, sa.Column("conversation_id", sa.String(), primary_key=True), sa.Column("user_id", sa.Integer()))
    memories = sa.Table("memories", metadata, sa.Column("id", sa.Integer(), primary_key=True), sa.Column("user_id", sa.Integer()),
        sa.Column("metadata_json", sa.JSON()), sa.Column("source_type", sa.String()), sa.Column("source_id", sa.String()))
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(users.insert(), [{"id": 1}, {"id": 2}])
        conn.execute(profiles.insert(), {"id": 1})
        conn.execute(convs.insert(), {"conversation_id": "owned", "user_id": 1})
        conn.execute(runs.insert(), {"id": 1, "user_id": 1, "conversation_id": "owned"})
        conn.execute(memories.insert(), [
            {"id": 1, "user_id": 1, "metadata_json": {"category": "answer_preference"}, "source_type": "", "source_id": ""},
            {"id": 2, "user_id": 1, "metadata_json": {"category": "tech_stack", "run_id": 1}, "source_type": "", "source_id": ""},
            {"id": 3, "user_id": 1, "metadata_json": {"category": "tech_stack"}, "source_type": "", "source_id": ""},
            {"id": 4, "user_id": 2, "metadata_json": {"category": "tech_stack", "conversation_id": "owned"}, "source_type": "", "source_id": ""},
        ])
        migration.op = Operations(MigrationContext.configure(conn))
        migration.upgrade()
        rows = conn.execute(sa.text("SELECT scope, scope_id FROM memories ORDER BY id")).all()
        assert rows == [("user", ""), ("conversation", "owned"), ("legacy_unscoped", ""), ("legacy_unscoped", "")]
        assert conn.execute(sa.text("SELECT use_memory, generate_memory FROM user_profiles")).one() == (1, 0)
        migration.downgrade()
        assert conn.execute(sa.text("SELECT COUNT(*) FROM memories")).scalar() == 4
        assert "scope" not in {c["name"] for c in sa.inspect(conn).get_columns("memories")}
