from src.web_app.graph.memory_projector import MemoryGraphProjector
from src.web_app.models.orm import User
from src.web_app.services.memory_service import MemoryService
from src.web_app.tests.db_test_utils import make_test_session


class FakeGraphRepository:
    def __init__(self):
        self.upserts = []
        self.status_updates = []

    def upsert_memory_projection(self, **kwargs):
        self.upserts.append(kwargs)

    def mark_memory_status(self, **kwargs):
        self.status_updates.append(kwargs)


def _user(db):
    user = User(email="graph@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_memory_graph_topic_is_user_scoped(monkeypatch):
    from src.web_app.core.config import settings

    monkeypatch.setattr(settings, "enable_neo4j", True)
    monkeypatch.setattr(settings, "neo4j_memory_graph_enabled", True)
    fake = FakeGraphRepository()
    projector = MemoryGraphProjector(repository=fake)

    projector.sync_memory(
        user_id=7,
        memory={
            "id": 10,
            "content": "用户正在做 Qdrant RAG 项目，偏好 best-effort 降级。",
            "memory_type": "semantic",
            "importance": 0.8,
            "metadata": {"category": "preference", "status": "active"},
        },
    )

    assert fake.upserts
    call = fake.upserts[0]
    assert call["user_id"] == 7
    assert call["memory"]["user_id"] == 7
    assert "qdrant" in call["topics"]
    assert call["preferences"]


def test_add_memory_disabled_neo4j_does_not_block_pg(monkeypatch):
    from src.web_app.core.config import settings

    monkeypatch.setattr(settings, "enable_neo4j", False)
    service = MemoryService()
    monkeypatch.setattr(service, "_get_qdrant_store", lambda: None)
    db = make_test_session()
    user = _user(db)

    result = service.add_memory(
        user.id,
        "用户偏好 Neo4j 只做关系索引，不保存完整正文。",
        "semantic",
        importance=0.8,
        metadata={"category": "preference", "status": "active"},
        db=db,
    )

    assert result["ok"] is True
    assert result["graph_indexed"] is False
    assert result["id"]


def test_forget_memory_marks_graph_only(monkeypatch):
    from src.web_app.core.config import settings

    monkeypatch.setattr(settings, "enable_neo4j", True)
    service = MemoryService()
    vector_deletes = []
    graph_marks = []
    monkeypatch.setattr(service, "_delete_memory_vector", lambda memory_id: vector_deletes.append(memory_id) or None)
    monkeypatch.setattr(service, "_sync_memory_graph", lambda user_id, memory: {"synced": True})
    monkeypatch.setattr(
        service,
        "_mark_memory_graph",
        lambda user_id, memory_id, status, reason: graph_marks.append((user_id, memory_id, status, reason)) or None,
    )
    db = make_test_session()
    user = _user(db)
    item = service.add_memory(
        user.id,
        "to forget",
        "semantic",
        importance=0.8,
        metadata={"category": "preference", "status": "active"},
        db=db,
    )

    result = service.forget_memory(user.id, item["id"], db=db)

    assert result["deleted"] == 1
    assert graph_marks == [(user.id, item["id"], "forgotten", "forget_memory")]
    assert vector_deletes == [item["id"]]

