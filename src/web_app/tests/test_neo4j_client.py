import pytest

from src.web_app.graph.neo4j_client import Neo4jClient, Neo4jUnavailable
from src.web_app.graph.schema import ensure_constraints


def test_neo4j_client_disabled(monkeypatch):
    from src.web_app.core.config import settings

    monkeypatch.setattr(settings, "enable_neo4j", False)
    monkeypatch.setattr(settings, "neo4j_uri", "")

    client = Neo4jClient()
    with pytest.raises(Neo4jUnavailable):
        client.get_driver()


def test_ensure_constraints_runs_schema_queries():
    class FakeClient:
        def __init__(self):
            self.queries = []

        def run_write(self, query, **params):
            self.queries.append((query, params))
            return []

    fake = FakeClient()
    ensure_constraints(fake)
    assert any("UserMemory" in query for query, _ in fake.queries)
    assert any("MemoryTopic" in query and "user_id" in query for query, _ in fake.queries)
    assert any("ProjectQdrantCollection" in query for query, _ in fake.queries)

