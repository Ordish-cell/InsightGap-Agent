from pathlib import Path
from typing import Any
import ast
import hashlib
import operator
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from src.web_app.db.repositories.artifact_repository import ArtifactRepository
from src.web_app.services.artifact_service import artifact_service
from src.web_app.services.memory_service import memory_service
from src.web_app.services.skill_service import skill_service
from src.web_app.mcp.local_file_tools import (
    local_file_append,
    local_file_delete,
    local_file_list,
    local_file_read,
    local_file_write,
)
from src.web_app.mcp.email_provider import MockEmailProvider, get_email_provider
from src.web_app.mcp.web_search_provider import web_search_provider

# Module-scope email provider instance (created once)
_email_provider = None


def _resolve_email_provider():
    global _email_provider
    if _email_provider is None:
        _email_provider = get_email_provider()
    return _email_provider


class LocalMCPProvider:
    def call(self, db: Session, user_id: int, tool_name: str, payload: dict[str, Any], agent_run_id: int | None = None) -> dict[str, Any]:
        handlers = {
            "search_mcp.search": self._search,
            "web.search": self._web_search,
            "system.time": self._system_time,
            "system.calc": self._system_calc,
            "system.unit_convert": self._system_unit_convert,
            "system.uuid": self._system_uuid,
            "system.hash": self._system_hash,
            "github_mcp.repo_summary": self._github_repo_summary,
            "file_mcp.read_artifact": self._read_artifact,
            "artifact_mcp.create_text_artifact": self._create_text_artifact,
            "memory_mcp.search": self._search_memory,
            "memory_mcp.add": self._add_memory,
            "skill_mcp.create_draft": self._create_skill_draft,
            "email_mcp.create_draft": self._create_email_draft,
            "email.send": self._send_email,
            "browser_mcp.plan_actions": self._plan_browser_actions,
            # ── Local file tools ────────────────────────────
            "local_file.list": self._list_local_files,
            "local_file.read": self._read_local_file,
            "local_file.write": self._write_local_file,
            "local_file.append": self._append_local_file,
            "local_file.delete": self._delete_local_file,
        }
        handler = handlers.get(tool_name)
        if not handler:
            raise ValueError("Tool not found")
        return handler(db, user_id, payload, agent_run_id)

    def _search(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        query = str(payload.get("query", "")).strip()
        limit = max(1, min(int(payload.get("limit", 5)), 10))
        return {
            "results": [
                {
                    "title": f"Local search result for {query or 'empty query'} #{idx + 1}",
                    "url": None,
                    "snippet": f"Deterministic local fallback result for query: {query}",
                    "source_type": "local_stub",
                }
                for idx in range(limit)
            ]
        }

    def _web_search(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        try:
            limit = int(payload.get("limit", 5) or 5)
        except (TypeError, ValueError):
            limit = 5
        return web_search_provider.search(
            str(payload.get("query", "")),
            limit=limit,
            recency_days=payload.get("recency_days"),
        )

    def _system_time(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        now = datetime.now().astimezone()
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekdays_zh = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return {
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "weekday": weekdays[now.weekday()],
            "weekday_zh": weekdays_zh[now.weekday()],
            "timezone": now.tzname() or str(now.astimezone().tzinfo),
            "utc_offset": now.strftime("%z"),
            "iso": now.isoformat(),
            "source": "local_system_clock",
        }

    def _system_calc(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        expression = str(payload.get("expression", "")).strip()
        if not expression:
            raise ValueError("Missing expression")
        result = _safe_eval_arithmetic(expression)
        return {"expression": expression, "result": result}

    def _system_unit_convert(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        value = float(payload.get("value"))
        from_unit = str(payload.get("from", "")).strip().lower()
        to_unit = str(payload.get("to", "")).strip().lower()
        result = _convert_unit(value, from_unit, to_unit)
        return {"value": value, "from": from_unit, "to": to_unit, "result": result}

    def _system_uuid(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        return {"uuid": str(uuid.uuid4())}

    def _system_hash(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        algorithm = str(payload.get("algorithm") or "sha256").strip().lower()
        text = str(payload.get("text") or "")
        if algorithm not in {"md5", "sha1", "sha256", "sha512"}:
            raise ValueError("Unsupported hash algorithm")
        digest = hashlib.new(algorithm)
        digest.update(text.encode("utf-8"))
        return {"algorithm": algorithm, "hash": digest.hexdigest()}

    def _github_repo_summary(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        repo = str(payload.get("repo", "unknown/repo")).strip() or "unknown/repo"
        return {"repo": repo, "summary": f"Deterministic local summary for repo {repo}", "signals": ["agent", "rag", "tooling"]}

    def _read_artifact(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        artifact_id = int(payload.get("artifact_id", 0))
        artifact = ArtifactRepository(db).get_by_user(user_id, artifact_id)
        if not artifact:
            raise ValueError("Artifact not found")
        path = Path(artifact.file_path)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        return {"artifact_id": artifact.id, "title": artifact.title, "artifact_type": artifact.artifact_type, "content": content}

    def _create_text_artifact(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        title = str(payload.get("title", "MCP Artifact")).strip() or "MCP Artifact"
        content = str(payload.get("content", ""))
        artifact_type = str(payload.get("artifact_type", "note")).strip() or "note"
        file_path = artifact_service.save_text_artifact(user_id, f"mcp_artifact_{agent_run_id or 'manual'}_{abs(hash(title))}.md", content)
        artifact = ArtifactRepository(db).create(user_id=user_id, run_id=agent_run_id, artifact_type=artifact_type, title=title, file_path=file_path, metadata_json={"source_type": "mcp_tool", "tool_name": "artifact_mcp.create_text_artifact"})
        return {"artifact_id": artifact.id, "title": artifact.title, "file_path": artifact.file_path}

    def _search_memory(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        query = str(payload.get("query", ""))
        limit = max(1, min(int(payload.get("limit", 5)), 20))
        return {"memories": memory_service.search_memory(user_id, query=query, db=db)[:limit]}

    def _add_memory(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        memory = memory_service.add_memory(user_id=user_id, content=str(payload.get("content", "")), memory_type=str(payload.get("memory_type", "episodic")), importance=float(payload.get("importance", 0.5)), metadata={"source_type": "mcp_tool", "agent_run_id": agent_run_id}, db=db)
        return {"memory_id": memory["id"], "memory": memory}

    def _create_skill_draft(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        draft = skill_service.create_skill_draft_from_run(agent_run_id or 0, user_id=user_id, db=db, payload=payload)
        return {"skill_id": draft["id"], "skill": draft}

    def _create_email_draft(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        return {"draft": {"to": payload.get("to", ""), "subject": payload.get("subject", ""), "body": payload.get("body", "")}, "sent": False}

    def _plan_browser_actions(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        url = payload.get("url") or "about:blank"
        return {"plan": [{"action": "open_url", "target": url}, {"action": "read_page", "target": url}, {"action": "summarize_for_goal", "target": payload.get("goal", "")}], "executed": False}

    # ── Email send ──────────────────────────────────────────────
    def _send_email(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        """Send email via the configured provider.

        Mock path: returns a sync result immediately — no event-loop hopping,
        no Future, no timeout.  This is a pure synchronous fast path.

        SMTP path: runs the async provider through the event loop.
        Timeouts are caught and returned as structured failures.
        """
        to = str(payload.get("to", ""))
        subject = str(payload.get("subject", ""))
        body = str(payload.get("body", ""))

        # ── Mock fast path (sync, never blocks) ──────────────────
        provider = _resolve_email_provider()
        if isinstance(provider, MockEmailProvider):
            return {
                "success": True,
                "provider": "mock",
                "sent": False,
                "message": "EMAIL_PROVIDER=mock, email was not actually sent.",
                "to": to,
                "subject": subject,
                "body": body,
                "body_preview": body[:200],
            }

        # ── SMTP path (needs event loop) ─────────────────────────
        import asyncio
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                # Running inside an event loop — schedule and wait
                import concurrent.futures
                future = concurrent.futures.Future()

                async def _send():
                    try:
                        result = await provider.send_email(to, subject, body)
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)

                loop.create_task(_send())
                try:
                    return future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    return {
                        "success": False,
                        "provider": "smtp",
                        "sent": False,
                        "to": to,
                        "subject": subject,
                        "body": body,
                        "body_preview": body[:200],
                        "message": "SMTP send timed out after 30s.",
                    }
            else:
                return asyncio.run(provider.send_email(to, subject, body))
        except Exception as exc:
            import logging
            _log = logging.getLogger(__name__)
            _log.exception("_send_email: SMTP failed")
            return {
                "success": False,
                "provider": "smtp",
                "sent": False,
                "to": to,
                "subject": subject,
                "body": body,
                "body_preview": body[:200],
                "message": f"SMTP send failed: {exc}",
            }

    # ── Local file tools ────────────────────────────────────────
    def _list_local_files(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        return local_file_list(path=str(payload.get("path", ".")))

    def _read_local_file(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        return local_file_read(path=str(payload.get("path", "")), max_chars=payload.get("max_chars"))

    def _write_local_file(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        return local_file_write(path=str(payload.get("path", "")), content=str(payload.get("content", "")), mode=str(payload.get("mode", "create_or_overwrite")))

    def _append_local_file(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        return local_file_append(path=str(payload.get("path", "")), content=str(payload.get("content", "")))

    def _delete_local_file(self, db: Session, user_id: int, payload: dict[str, Any], agent_run_id: int | None) -> dict[str, Any]:
        return local_file_delete(path=str(payload.get("path", "")))


local_provider = LocalMCPProvider()


_ARITHMETIC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_arithmetic(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_OPERATORS:
            left = _eval(node.left)
            right = _eval(node.right)
            return _ARITHMETIC_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC_OPERATORS:
            return _ARITHMETIC_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Expression contains unsupported syntax")

    result = _eval(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


_LINEAR_UNIT_FACTORS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "mile": 1609.344,
    "miles": 1609.344,
    "mi": 1609.344,
    "gb": 1024.0,
    "mb": 1.0,
    "kb": 1 / 1024,
    "kg": 1.0,
    "g": 0.001,
    "lb": 0.45359237,
    "lbs": 0.45359237,
}


def _convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit in {"c", "celsius", "摄氏", "摄氏度"} and to_unit in {"f", "fahrenheit", "华氏", "华氏度"}:
        return round((value * 9 / 5) + 32, 6)
    if from_unit in {"f", "fahrenheit", "华氏", "华氏度"} and to_unit in {"c", "celsius", "摄氏", "摄氏度"}:
        return round((value - 32) * 5 / 9, 6)
    if from_unit not in _LINEAR_UNIT_FACTORS or to_unit not in _LINEAR_UNIT_FACTORS:
        raise ValueError("Unsupported unit conversion")
    base_value = value * _LINEAR_UNIT_FACTORS[from_unit]
    return round(base_value / _LINEAR_UNIT_FACTORS[to_unit], 6)
