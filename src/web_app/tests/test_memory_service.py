from src.web_app.models.orm import User
from src.web_app.services.memory_service import MemoryService
from src.web_app.tests.db_test_utils import make_test_session


def test_memory_add_search_summary_masks_sensitive():
    db = make_test_session()
    user = User(email="memory@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)

    service = MemoryService()
    service.add_memory(user.id, "normal note", "working", 0.5, {}, db)
    service.add_memory(user.id, "sensitive note", "working", 0.9, {"sensitive": True}, db)

    assert service.search_memory(user.id, "note", db=db)
    summary = service.summarize_memory(user.id, db)
    assert "sensitive note" not in str(summary)
    assert "masked sensitive memory" in str(summary)
