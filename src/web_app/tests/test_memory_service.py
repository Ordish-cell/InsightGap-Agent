from src.web_app.models.orm import User
from src.web_app.services.memory_service import MemoryService
from src.web_app.tests.db_test_utils import make_test_session


class FakeQdrantStore:
    def __init__(self):
        self.upserts = []

    def upsert_memory(self, **kwargs):
        self.upserts.append(kwargs)
        return f"point-{kwargs['memory_id']}"


def _user(db, email="memory@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_memory_add_search_summary_masks_sensitive():
    db = make_test_session()
    user = _user(db)

    service = MemoryService()
    service.add_memory(user.id, "normal note", "working", 0.5, {}, db)
    service.add_memory(user.id, "sensitive note", "working", 0.9, {"sensitive": True}, db)

    assert service.search_memory(user.id, "note", db=db)
    summary = service.summarize_memory(user.id, db)
    assert "sensitive note" not in str(summary)
    assert "masked sensitive memory" in str(summary)


def test_working_memory_does_not_initialize_or_write_qdrant(monkeypatch):
    db = make_test_session()
    user = _user(db, "working-qdrant@example.com")
    service = MemoryService()

    def fail_qdrant_init():
        raise AssertionError("working memory must not touch Qdrant")

    monkeypatch.setattr(service, "_get_qdrant_store", fail_qdrant_init)
    monkeypatch.setattr(service, "_sync_memory_graph", lambda *_args, **_kwargs: {"synced": False})

    result = service.add_memory(
        user.id,
        "temporary page context",
        memory_type="working",
        importance=0.9,
        metadata={"visible_in_long_term_memory": False},
        db=db,
    )

    assert result["ok"] is True
    assert result["memory_type"] == "working"
    assert result["qdrant_indexed"] is False
    assert result["qdrant_point_id"] is None
    assert result["error"] is None


def test_long_term_memories_write_qdrant(monkeypatch):
    db = make_test_session()
    user = _user(db, "long-term-qdrant@example.com")
    service = MemoryService()
    store = FakeQdrantStore()

    monkeypatch.setattr(service, "_get_qdrant_store", lambda: store)
    monkeypatch.setattr(service, "_sync_memory_graph", lambda *_args, **_kwargs: {"synced": False})

    semantic = service.add_memory(
        user.id,
        "user prefers concise answers",
        memory_type="semantic",
        importance=0.9,
        metadata={"visible_in_long_term_memory": True},
        db=db,
    )
    episodic = service.add_memory(
        user.id,
        "user completed a research workflow",
        memory_type="episodic",
        importance=0.86,
        metadata={"visible_in_long_term_memory": True},
        db=db,
    )

    assert semantic["qdrant_indexed"] is True
    assert episodic["qdrant_indexed"] is True
    assert [call["memory_type"] for call in store.upserts] == ["semantic", "episodic"]


def test_extracted_working_memory_is_not_counted_as_qdrant_indexed(monkeypatch):
    db = make_test_session()
    user = _user(db, "extracted-working@example.com")
    service = MemoryService()
    store = FakeQdrantStore()

    monkeypatch.setattr(service, "_get_qdrant_store", lambda: store)
    monkeypatch.setattr(service, "_sync_memory_graph", lambda *_args, **_kwargs: {"synced": False})

    result = service._save_extracted(
        user.id,
        {
            "working_memories": [
                {"content": "selected feed card: vector bug", "importance": 0.7, "category": "working_context"}
            ],
            "episodic_memories": [],
            "semantic_memories": [],
            "should_consolidate": False,
        },
        db,
        "test-run",
    )

    assert result["saved"]["working"][0]["qdrant_indexed"] is False
    assert result["qdrant_indexed_count"] == 0
    assert store.upserts == []
