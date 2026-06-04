import re
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.db.repositories.skill_repository import SkillRepository

REUSABLE_INTENT_TERMS = ("以后", "下次", "复用", "流程", "保存成", "生成 skill", "create skill", "monitor", "监控", "自动提醒")
WORKFLOW_TERMS = ("研究", "报告", "artifact", "文档", "总结", "对比", "分析", "生成", "tool", "mcp", "rag", "research")


class SkillService:
    def create_skill_draft_from_run(self, run_id: int, user_id: int = 1, db: Session | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = payload or {}
        if db:
            item = SkillRepository(db).create(
                user_id=user_id,
                name=data.get("name") or data.get("title") or f"Skill draft from run {run_id}",
                description=data.get("description", ""),
                trigger_text=data.get("trigger_text") or "\n".join(data.get("trigger_patterns", [])),
                input_schema=data.get("input_schema", {}),
                context_recipe=data.get("context_recipe") or data.get("workflow_steps", []),
                tool_plan=data.get("tool_plan") or data.get("required_tools", []),
                output_schema=data.get("output_schema") or data.get("output_contract", {}),
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

    def match_skill(self, user_input: str, user_id: int, db: Session | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not db:
            return {"matched": False, "user_id": user_id, "input": user_input, "candidates": []}
        query_terms = self._terms(" ".join([user_input, self._context_text(context or {})]))
        candidates = []
        for skill in SkillRepository(db).list_by_user(user_id):
            if skill.status == "disabled":
                continue
            score, reason = self._score_skill(query_terms, skill)
            if score >= 0.5:
                item = {**self._to_dict(skill), "match_score": score, "match_reason": reason, "auto_use": skill.status == "approved" and score >= 0.75}
                candidates.append(item)
        candidates.sort(key=lambda item: item["match_score"], reverse=True)
        matched = candidates[0] if candidates and candidates[0]["auto_use"] else None
        return {"matched": bool(matched), "matched_skill": matched, "candidate_skills": candidates[:3], "user_id": user_id, "input": user_input}

    def evaluate_reusability(self, state: dict[str, Any]) -> dict[str, Any]:
        text = str(state.get("user_input", "")).lower()
        route = state.get("route")
        artifacts = state.get("artifacts", [])
        tool_call = state.get("tool_call") or {}
        successful = state.get("status") not in {"failed", "waiting_approval"} and not state.get("error")
        user_intent = 1.0 if any(term in text for term in REUSABLE_INTENT_TERMS) else 0.0
        workflow_structure = 1.0 if route in {"research", "rag", "artifact", "tool", "skill"} or any(term in text for term in WORKFLOW_TERMS) else 0.0
        artifact_output = 1.0 if artifacts or route in {"research", "artifact"} else 0.0
        tool_chain = 1.0 if tool_call or route in {"research", "rag", "tool"} else 0.0
        repeatability = 1.0 if user_intent or workflow_structure else 0.0
        score = round(0.30 * repeatability + 0.20 * workflow_structure + 0.15 * artifact_output + 0.15 * tool_chain + 0.10 * user_intent + 0.10 * float(successful), 2)
        reasons = []
        if user_intent:
            reasons.append("user_intent_signal")
        if workflow_structure:
            reasons.append("workflow_structure")
        if artifact_output:
            reasons.append("artifact_output")
        if tool_chain:
            reasons.append("tool_chain_specificity")
        if successful:
            reasons.append("successful_completion")
        return {"reusable_score": score, "should_create": score >= 0.70, "should_suggest": 0.50 <= score < 0.70, "reason": ", ".join(reasons) or "low_reuse_signal"}

    def _to_dict(self, item) -> dict[str, Any]:
        return {"id": item.id, "user_id": item.user_id, "name": item.name, "description": item.description, "trigger_text": item.trigger_text, "input_schema": item.input_schema, "context_recipe": item.context_recipe, "tool_plan": item.tool_plan, "output_schema": item.output_schema, "safety_level": item.safety_level, "status": item.status, "version": item.version}

    def _score_skill(self, query_terms: set[str], skill) -> tuple[float, str]:
        skill_terms = self._terms(" ".join([skill.name, skill.description, skill.trigger_text, " ".join(map(str, skill.context_recipe or [])), " ".join(map(str, skill.tool_plan or []))]))
        if not query_terms or not skill_terms:
            return 0.0, "no comparable terms"
        overlap = query_terms & skill_terms
        trigger_pattern_match = 1.0 if self._trigger_phrase_matches(query_terms, skill.trigger_text) else min(1.0, len(overlap) / max(1, min(len(query_terms), len(skill_terms))))
        semantic_similarity = len(overlap) / max(1, len(query_terms | skill_terms))
        task_type_match = 1.0 if {"研究", "research", "报告", "artifact", "skill", "监控"} & overlap else min(1.0, len(overlap) / 3)
        recency_or_success_score = 0.5 if skill.status == "approved" else 0.0
        score = round(0.35 * trigger_pattern_match + 0.25 * semantic_similarity + 0.20 * task_type_match + 0.10 * 0.0 + 0.10 * recency_or_success_score, 2)
        if skill.status == "approved" and trigger_pattern_match >= 1.0:
            score = max(score, 0.80)
        return score, f"matched terms: {', '.join(sorted(overlap)[:8])}" if overlap else "no shared trigger terms"

    def _trigger_phrase_matches(self, query_terms: set[str], trigger_text: str) -> bool:
        phrases = [part.strip().lower() for part in re.split(r"[\n,，;；]+", trigger_text or "") if part.strip()]
        return any(self._terms(phrase) and self._terms(phrase).issubset(query_terms) for phrase in phrases)

    def _terms(self, text: str) -> set[str]:
        lowered = text.lower()
        words = set(re.findall(r"[a-z0-9_]+", lowered))
        words.update(term for term in ("研究", "报告", "文档", "流程", "复用", "监控", "信息差", "知识库", "总结", "对比", "分析", "提醒") if term in lowered)
        return {word for word in words if len(word) > 1}

    def _context_text(self, context: dict[str, Any]) -> str:
        feed_card = context.get("feed_card") or {}
        return " ".join(str(feed_card.get(key, "")) for key in ("title", "one_sentence_value", "why_you", "information_gap", "summary", "domain", "source_type"))

    # ── Skill Evolution ───────────────────────────────────────────────

    def record_skill_usage(self, skill_id: int, user_id: int, success: bool, db: Session | None = None) -> dict[str, Any]:
        """Record a skill usage event and update evolution stats."""
        if not db:
            return {"skill_id": skill_id, "recorded": False}
        repo = SkillRepository(db)
        skill = repo.get_by_user(user_id, skill_id)
        if not skill:
            return {"skill_id": skill_id, "error": "not_found"}

        stats = self._get_evolution_stats(skill)
        if success:
            stats["success_count"] = stats.get("success_count", 0) + 1
        else:
            stats["failure_count"] = stats.get("failure_count", 0) + 1

        total = stats["success_count"] + stats["failure_count"]
        stats["confidence"] = round(stats["success_count"] / max(1, total), 2)
        stats["last_used_at"] = __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()

        # Store evolution stats in eval_checks
        updated_checks = self._merge_evolution_into_checks(skill.eval_checks or [], stats)
        repo.update(skill, eval_checks=updated_checks)
        return {"skill_id": skill_id, "stats": stats}

    def get_skill_evolution(self, skill_id: int, user_id: int, db: Session | None = None) -> dict[str, Any]:
        """Get evolution stats for a skill."""
        if not db:
            return {}
        repo = SkillRepository(db)
        skill = repo.get_by_user(user_id, skill_id)
        if not skill:
            return {}
        return self._get_evolution_stats(skill)

    def detect_repeated_workflow(
        self, user_id: int, user_input: str, db: Session | None = None
    ) -> dict[str, Any]:
        """Check if the current task is similar to previously completed tasks.
        Returns a reusable_score boost if a repeated pattern is detected."""
        if not db:
            return {"repeated": False, "boost": 0.0}

        skills = SkillRepository(db).list_by_user(user_id)
        input_terms = self._terms(user_input)

        best_score = 0.0
        best_skill = None
        for skill in skills:
            if skill.status in ("disabled",):
                continue
            skill_terms = self._terms(" ".join([
                skill.name, skill.description, skill.trigger_text,
                " ".join(map(str, skill.context_recipe or [])),
            ]))
            if not input_terms or not skill_terms:
                continue
            overlap = len(input_terms & skill_terms)
            similarity = overlap / max(1, len(input_terms | skill_terms))
            if similarity > best_score:
                best_score = similarity
                best_skill = self._to_dict(skill)

        stats = self._get_evolution_stats(best_skill) if best_skill else {}
        success_count = stats.get("success_count", 0)

        # Boost reusable_score if a repeated pattern is found with >= 2 prior successes
        boost = 0.0
        if best_score >= 0.50 and success_count >= 2:
            boost = min(0.25, 0.10 * success_count)

        return {
            "repeated": boost > 0,
            "boost": round(boost, 2),
            "similar_skill_id": best_skill.get("id") if best_skill else None,
            "similarity": round(best_score, 2),
            "prior_success_count": success_count,
        }

    def _get_evolution_stats(self, skill) -> dict[str, Any]:
        """Extract evolution stats from skill's eval_checks."""
        checks = getattr(skill, "eval_checks", None) or []
        for item in checks:
            if isinstance(item, dict) and item.get("_type") == "skill_evolution":
                return {
                    "success_count": item.get("success_count", 0),
                    "failure_count": item.get("failure_count", 0),
                    "confidence": item.get("confidence", 0.5),
                    "last_used_at": item.get("last_used_at", ""),
                }
        return {"success_count": 0, "failure_count": 0, "confidence": 0.5, "last_used_at": ""}

    def _merge_evolution_into_checks(self, checks: list, stats: dict) -> list:
        """Merge evolution stats into the eval_checks list."""
        new_checks = []
        found = False
        for item in checks:
            if isinstance(item, dict) and item.get("_type") == "skill_evolution":
                new_checks.append({"_type": "skill_evolution", **stats})
                found = True
            else:
                new_checks.append(item)
        if not found:
            new_checks.append({"_type": "skill_evolution", **stats})
        return new_checks


skill_service = SkillService()
