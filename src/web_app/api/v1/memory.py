from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
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
