from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.web_app.db.repositories.memory_repository import MemoryRepository
from src.web_app.db.session import get_db
from src.web_app.memory.qdrant_memory_store import QdrantMemoryStore
from src.web_app.schemas.common import fail, ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.memory_service import memory_service
from src.web_app.services.user_growth_service import user_growth_service

router = APIRouter()


@router.post("/add")
def add_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.add_memory(user_id, payload.get("content", ""), payload.get("memory_type", "working"), payload.get("importance", 0.0), payload.get("metadata", {}), db))


@router.post("/search")
def search_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.search_memory(user_id, payload.get("query", ""), payload.get("memory_type"), payload.get("min_importance", 0.0), db))


@router.post("/consolidate")
def consolidate_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.consolidate_memory(user_id, db))


@router.post("/forget")
def forget_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.forget_memory(user_id, payload.get("memory_id"), db))


@router.get("/summary")
def memory_summary(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(memory_service.summarize_memory(user_id, db))


@router.post("/reflect")
def reflect_memory(payload: dict, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(user_growth_service.reflect_user_profile(user_id, db))


@router.get("/growth-profile")
def growth_profile(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    """Return the user's dynamic growth profile — categorized long-term settings
    with effective importance, grouped by category, with status indicators."""
    all_semantic = user_growth_service.get_memories_with_effective_importance(
        user_id, db, memory_type="semantic", min_effective=0.0,
    )
    all_episodic = user_growth_service.get_memories_with_effective_importance(
        user_id, db, memory_type="episodic", min_effective=0.0,
    )[:10]

    # Group semantic by category
    by_category: dict[str, list[dict]] = {}
    for mem in all_semantic:
        meta = mem.get("metadata", {})
        cat = meta.get("category", "uncategorized")
        status = meta.get("status", "active")
        by_category.setdefault(cat, []).append({
            "id": mem.get("id"),
            "content": mem.get("content"),
            "importance": mem.get("importance"),
            "effective_importance": mem.get("effective_importance"),
            "status": status,
            "stability": meta.get("stability", ""),
            "confidence": meta.get("confidence", 0),
            "evidence_count": meta.get("evidence_count", 1),
            "last_seen_at": meta.get("last_seen_at", ""),
            "superseded_by": meta.get("superseded_by"),
            "supersedes": meta.get("supersedes"),
        })

    category_labels: dict[str, str] = {
        "project_goal": "项目目标",
        "project_goal_summary": "项目目标（总结）",
        "tech_stack": "技术栈",
        "tech_stack_summary": "技术栈（总结）",
        "preference": "产品偏好",
        "ui_preference": "UI 偏好",
        "ui_preference_summary": "UI 偏好（总结）",
        "product_principle": "产品原则",
        "boundary": "当前边界",
        "boundary_summary": "当前边界（总结）",
        "feed_interest": "信息兴趣",
        "feed_interest_summary": "信息兴趣（总结）",
        "negative_preference": "负面偏好",
        "workflow_pattern": "任务模式",
        "workflow_pattern_summary": "任务模式（总结）",
        "skill_preference": "Skill 偏好",
        "research_preference": "研究偏好",
        "artifact_preference": "Artifact 偏好",
        "temporary_task": "临时任务",
        "episodic_feedback": "行为反馈",
        "uncategorized": "其他",
    }

    categories = []
    for cat, mems in sorted(by_category.items()):
        categories.append({
            "category": cat,
            "label": category_labels.get(cat, cat),
            "memories": sorted(mems, key=lambda m: m["effective_importance"], reverse=True),
            "count": len(mems),
        })

    return ok({
        "categories": categories,
        "semantic_count": len(all_semantic),
        "episodic_count": len(all_episodic),
        "recent_episodic": all_episodic[:5],
        "dynamic_profile": user_growth_service.build_dynamic_preference_profile(user_id, db),
    })


def _enrich_long_term_memory(m) -> dict:
    d = memory_service._to_dict(m)
    meta = m.metadata_json or {}
    d["category"] = meta.get("category", "")
    d["confidence"] = meta.get("confidence", 0)
    d["stability"] = meta.get("stability", "")
    d["status"] = meta.get("status", "active")
    d["evidence_count"] = meta.get("evidence_count", 1)
    d["last_seen_at"] = meta.get("last_seen_at", "")
    d["visible_in_long_term_memory"] = meta.get("visible_in_long_term_memory", False)
    d["effective_importance"] = user_growth_service.compute_effective_importance(d)
    d["created_at"] = str(m.created_at) if m.created_at else None
    d["updated_at"] = str(m.updated_at) if m.updated_at else None
    return d

@router.get("/long-term")
def list_long_term_memories(user_id=Depends(get_current_user_id), db=Depends(get_db), memory_type=Query(None, alias="type"), category=Query(None), status=Query(None), query=Query(None), page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    repo = MemoryRepository(db)
    items, total = repo.list_long_term(user_id, memory_type=memory_type, category=category, status=status, query=query, page=int(page), page_size=int(page_size))
    return ok({"items": [_enrich_long_term_memory(m) for m in items], "total": total, "page": page, "page_size": page_size})

@router.post("/long-term/search")
def search_long_term(payload: dict, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    repo = MemoryRepository(db)
    query = str(payload.get("query", "") or payload.get("q", ""))
    memory_type = payload.get("type") or payload.get("memory_type")
    items, total = repo.list_long_term(user_id, memory_type=memory_type, category=payload.get("category"), status=payload.get("status"), query=query, page=int(payload.get("page", 1)), page_size=int(payload.get("page_size", 20)))
    return ok({"items": [_enrich_long_term_memory(m) for m in items], "total": total, "page": int(payload.get("page", 1)), "page_size": int(payload.get("page_size", 20)), "query": query})

@router.patch("/{memory_id}")
def update_memory(memory_id: int, payload: dict, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    import logging as _log
    repo = MemoryRepository(db)
    item = repo.get_by_id(memory_id)
    if not item or item.user_id != user_id: return fail("not_found", f"Memory {memory_id} not found")
    changed = False
    if "importance" in payload:
        importance = float(payload["importance"])
        if 0.0 <= importance <= 1.0: repo.update(item, importance=importance); changed = True
    if "status" in payload:
        meta = dict(item.metadata_json or {}); meta["status"] = payload["status"]
        repo.update(item, metadata_json=meta); changed = True
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        meta = dict(item.metadata_json or {}); meta.update(payload["metadata"])
        repo.update(item, metadata_json=meta); changed = True
    if changed and item.qdrant_point_id:
        try:
            store = QdrantMemoryStore(); db.refresh(item)
            store.upsert_memory(memory_id=item.id, user_id=item.user_id, content=item.content, memory_type=item.memory_type, importance=item.importance, source_type=item.source_type or "", metadata=item.metadata_json or {}, point_id=item.qdrant_point_id)
        except Exception: _log.warning("qdrant_sync_failed", exc_info=True)
    db.refresh(item)
    return ok(_enrich_long_term_memory(item))

@router.delete("/{memory_id}")
def delete_memory(memory_id: int, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    repo = MemoryRepository(db)
    item = repo.get_by_id(memory_id)
    if not item or item.user_id != user_id: return fail("not_found", f"Memory {memory_id} not found")
    result = memory_service.forget_memory(user_id, memory_id, db)
    return ok({"deleted": bool(result.get("deleted")), "memory_id": memory_id, **({"vector_cleanup_warning": result["vector_cleanup_warning"]} if result.get("vector_cleanup_warning") else {})})

@router.post("/{memory_id}/archive")
def archive_memory(memory_id: int, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    repo = MemoryRepository(db)
    item = repo.get_by_id(memory_id)
    if not item or item.user_id != user_id: return fail("not_found", f"Memory {memory_id} not found")
    meta = dict(item.metadata_json or {}); meta["status"] = "archived"
    repo.update(item, metadata_json=meta)
    return ok({"memory_id": memory_id, "status": "archived"})

@router.post("/{memory_id}/restore")
def restore_memory(memory_id: int, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    repo = MemoryRepository(db)
    item = repo.get_by_id(memory_id)
    if not item or item.user_id != user_id: return fail("not_found", f"Memory {memory_id} not found")
    meta = dict(item.metadata_json or {}); meta["status"] = "active"
    repo.update(item, metadata_json=meta)
    return ok({"memory_id": memory_id, "status": "active"})

@router.post("/forget/by-importance")
def forget_by_importance(payload: dict, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    threshold = float(payload.get("threshold", 0.2))
    memory_type = payload.get("memory_type")
    return ok(memory_service.forget_by_importance(user_id, threshold=threshold, memory_type=memory_type, db=db))

@router.post("/forget/by-time")
def forget_by_time(payload: dict, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    max_age_days = int(payload.get("max_age_days", 90))
    memory_type = payload.get("memory_type")
    return ok(memory_service.forget_by_time(user_id, max_age_days=max_age_days, memory_type=memory_type, db=db))

@router.post("/forget/by-capacity")
def forget_by_capacity(payload: dict, user_id=Depends(get_current_user_id), db=Depends(get_db)):
    memory_type = payload.get("memory_type", "semantic")
    max_capacity = int(payload.get("max_capacity", 500))
    return ok(memory_service.forget_by_capacity(user_id, memory_type=memory_type, max_capacity=max_capacity, db=db))
