from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any

from src.web_app.core.config import settings
from src.web_app.graph.neo4j_client import Neo4jUnavailable
from src.web_app.graph.repositories import GraphRepository

logger = logging.getLogger(__name__)

_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-./#]{1,40}|[\u4e00-\u9fff]{2,12}")


class GraphContextService:
    def __init__(self, repository: GraphRepository | None = None):
        self.repository = repository or GraphRepository()
        self.last_debug: dict[str, Any] = {}

    def get_context(self, *, user_id: int, query: str, route: str = "chat", limit: int | None = None) -> str:
        if not self._enabled():
            self.last_debug = {"enabled": False, "reason": "disabled"}
            return ""
        started = perf_counter()
        terms = _query_terms(query)
        is_project_diagnostic = _is_project_diagnostic_query(query)
        limit = limit or settings.neo4j_graph_context_limit
        try:
            memory_rows = [] if is_project_diagnostic else self.repository.get_user_memory_context(user_id=user_id, terms=terms, limit=limit)
            project_rows = self.repository.get_project_context(
                project_key=settings.neo4j_project_key,
                terms=terms,
                limit=max(limit, 12) if is_project_diagnostic else limit,
            )
            text = self._format(memory_rows, project_rows, query=query, is_project_diagnostic=is_project_diagnostic)
            self.last_debug = {
                "enabled": True,
                "memory_rows": len(memory_rows),
                "project_rows": len(project_rows),
                "terms": terms,
                "project_diagnostic": is_project_diagnostic,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "route": route,
            }
            return text
        except Neo4jUnavailable as exc:
            logger.warning("graph.context_unavailable user_id=%s reason=%s", user_id, exc)
            self.last_debug = {"enabled": True, "warning": str(exc), "fallback": True}
            return ""
        except Exception as exc:
            logger.warning("graph.context_failed user_id=%s error=%s", user_id, exc, exc_info=True)
            self.last_debug = {"enabled": True, "warning": str(exc)[:200], "fallback": True}
            return ""

    def _enabled(self) -> bool:
        return bool(settings.enable_neo4j and settings.neo4j_context_enabled)

    def _format(
        self,
        memory_rows: list[dict[str, Any]],
        project_rows: list[dict[str, Any]],
        *,
        query: str = "",
        is_project_diagnostic: bool = False,
    ) -> str:
        lines: list[str] = []
        if is_project_diagnostic:
            lines.append("## Project Diagnostic Map")
            lines.append("- 文档上传/入库失败优先排查：API 路由、DocumentService、document_parser、structured_chunker/chunker、embeddings、QdrantVectorStore、DocumentRepository。")
            lines.append("- 如果错误发生在上传后异步/同步 ingest，先看 DocumentService 的状态流转和 failed_stage/error_message。")
            lines.append("- 如果错误包含 DashScope/embedding，先看 embeddings.py 的 batch、输入长度、模型名和 response body。")
            lines.append("- 如果错误包含 Qdrant/vector/upsert/delete，先看 vector_store.py、qdrant_client.py 和 collection 配置。")
        if memory_rows:
            lines.append("## User Memory Graph")
            seen: set[str] = set()
            for row in memory_rows:
                memory_id = str(row.get("memory_id", ""))
                target = row.get("target_key") or ""
                key = f"{memory_id}:{target}"
                if key in seen:
                    continue
                seen.add(key)
                category = row.get("category") or ""
                preview = row.get("preview") or ""
                target_text = f" -> {target}" if target else ""
                lines.append(f"- memory#{memory_id} [{category}]{target_text}: {preview}")
        if project_rows:
            lines.append("## Project Knowledge Graph")
            for row in project_rows:
                labels = row.get("labels") or []
                label = labels[0] if labels else "ProjectNode"
                key = row.get("key") or row.get("name") or ""
                path = row.get("path") or ""
                suffix = f" ({path})" if path else ""
                lines.append(f"- {label}: {key}{suffix}")
        return "\n".join(lines)


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _TERM_RE.findall(query or ""):
        term = raw.strip().lower()
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    for alias in _diagnostic_aliases(query):
        if alias not in seen:
            seen.add(alias)
            terms.append(alias)
    return terms[:12]


def _is_project_diagnostic_query(query: str) -> bool:
    text = (query or "").lower()
    has_failure = any(term in text for term in (
        "\u5931\u8d25",  # 失败
        "\u62a5\u9519",  # 报错
        "\u9519\u8bef",  # 错误
        "\u6392\u67e5",  # 排查
        "\u770b\u54ea\u4e9b\u6a21\u5757",  # 看哪些模块
        "\u54ea\u4e9b\u6a21\u5757",  # 哪些模块
        "diagnostic", "troubleshoot", "failed", "error",
    ))
    has_project_area = any(term in text for term in (
        "\u4e0a\u4f20",  # 上传
        "\u6587\u6863",  # 文档
        "\u6587\u4ef6",  # 文件
        "\u5165\u5e93",  # 入库
        "\u6444\u5165",  # 摄入
        "embedding", "qdrant", "rag", "upload", "document", "ingest",
    ))
    return has_failure and has_project_area


def _diagnostic_aliases(query: str) -> list[str]:
    text = (query or "").lower()
    aliases: list[str] = []
    if _is_project_diagnostic_query(text):
        aliases.extend([
            "document",
            "documents",
            "upload",
            "ingest",
            "document_service",
            "parser",
            "chunk",
            "embedding",
            "embeddings",
            "qdrant",
            "vector",
            "repository",
        ])
    return aliases


graph_context_service = GraphContextService()
