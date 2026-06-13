from pathlib import Path

from src.web_app.graph.project_projector import ProjectGraphBuilder
from src.web_app.graph.repositories import GraphRepository


class FakeClient:
    def __init__(self):
        self.writes = []

    def run_write(self, query, **params):
        self.writes.append((query, params))
        return []

    def run_read(self, query, **params):
        return []


def test_project_graph_builder_extracts_rule_based_nodes(tmp_path: Path):
    api_dir = tmp_path / "src/web_app/api/v1"
    service_dir = tmp_path / "src/web_app/services"
    repo_dir = tmp_path / "src/web_app/db/repositories"
    rag_dir = tmp_path / "src/web_app/rag"
    for path in (api_dir, service_dir, repo_dir, rag_dir):
        path.mkdir(parents=True, exist_ok=True)
    (api_dir / "documents.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.post('/documents/chat-upload')\n"
        "def upload(): pass\n",
        encoding="utf-8",
    )
    (service_dir / "document_service.py").write_text(
        "from src.web_app.core.config import settings\n"
        "class DocumentService:\n"
        "    def ingest(self): return settings.qdrant_hybrid_collection\n",
        encoding="utf-8",
    )
    (repo_dir / "document_repository.py").write_text(
        "class DocumentRepository:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (rag_dir / "vector_store.py").write_text(
        "COLLECTION = 'agent_os_documents_v3'\n"
        "MEM = 'memory_vectors'\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("NEO4J_URI=x\nNEO4J_PASSWORD=secret\nQDRANT_URL=y\n", encoding="utf-8")

    result = ProjectGraphBuilder(root=tmp_path, project_key="agent_os_test").build()

    assert result.summary["api_endpoints"] == 1
    assert result.summary["services"] == 1
    assert result.summary["repositories"] == 1
    assert result.summary["qdrant_collections"] >= 2
    assert any(item["name"] == "agent_os_documents_v3" for item in result.graph["qdrant_collections"])
    assert any(item["name"] == "memory_vectors" for item in result.graph["qdrant_collections"])
    assert any(item["key"] == "qdrant_hybrid_collection" for item in result.graph["config_keys"])
    assert not any(item["key"] == "NEO4J_PASSWORD" for item in result.graph["config_keys"])


def test_project_graph_repository_writes_groups_without_empty_list_loss():
    fake = FakeClient()
    repo = GraphRepository(client=fake)
    graph = {
        "project_key": "agent_os",
        "project_name": "Agent OS",
        "modules": [{"key": "src.web_app.services.document_service", "props": {"key": "src.web_app.services.document_service", "project_key": "agent_os"}}],
        "services": [],
        "repositories": [],
        "api_endpoints": [],
        "config_keys": [],
        "qdrant_collections": [{"name": "agent_os_documents_v3", "props": {"name": "agent_os_documents_v3", "key": "agent_os_documents_v3", "project_key": "agent_os"}}],
        "technologies": [],
        "edges": [],
    }

    repo.upsert_project_graph(graph)

    assert any("Project " in query for query, _ in fake.writes)
    assert any("ProjectModule" in query for query, _ in fake.writes)
    assert any("ProjectQdrantCollection" in query for query, _ in fake.writes)
