from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.skill_repository import SkillRepository


class SkillService:
    def create_skill_draft_from_run(self, run_id: int, user_id: int = 1, db: Session | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = payload or {}
        if db:
            item = SkillRepository(db).create(
                user_id=user_id,
                name=data.get("name", f"Skill draft from run {run_id}"),
                description=data.get("description", ""),
                trigger_text=data.get("trigger_text", ""),
                input_schema=data.get("input_schema", {}),
                context_recipe=data.get("context_recipe", []),
                tool_plan=data.get("tool_plan", []),
                output_schema=data.get("output_schema", {}),
                safety_level=data.get("safety_level", "read_only"),
                eval_checks=data.get("eval_checks", []),
            )
            return self._to_dict(item)
        return {"run_id": run_id, "user_id": user_id, "status": "draft", "safety_level": "read_only", "tool_plan": []}

    def list_skills(self, user_id: int, db: Session | None = None) -> list[dict[str, Any]]:
        if db:
            return [self._to_dict(item) for item in SkillRepository(db).list_by_user(user_id)]
        return []

    def approve_skill(self, skill_id: int, user_id: int = 1, db: Session | None = None) -> dict[str, Any]:
        if db:
            repo = SkillRepository(db)
            item = repo.get_by_user(user_id, skill_id)
            if not item:
                raise ValueError("Skill not found")
            return self._to_dict(repo.update(item, status="approved"))
        return {"skill_id": skill_id, "status": "approved"}

    def disable_skill(self, skill_id: int, user_id: int = 1, db: Session | None = None) -> dict[str, Any]:
        if db:
            repo = SkillRepository(db)
            item = repo.get_by_user(user_id, skill_id)
            if not item:
                raise ValueError("Skill not found")
            return self._to_dict(repo.update(item, status="disabled"))
        return {"skill_id": skill_id, "status": "disabled"}

    def match_skill(self, user_input: str, user_id: int) -> dict[str, Any]:
        return {"matched": False, "user_id": user_id, "input": user_input}

    def _to_dict(self, item) -> dict[str, Any]:
        return {"id": item.id, "user_id": item.user_id, "name": item.name, "description": item.description, "trigger_text": item.trigger_text, "safety_level": item.safety_level, "status": item.status, "version": item.version}


skill_service = SkillService()
