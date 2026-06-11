"""Compare RAG retrieval backends without writing data.

Examples:
    python scripts/compare_rag_backends.py --user-id 2 --query "合同号 HT-2026-001"
    python scripts/compare_rag_backends.py --user-id 2 --queries-file eval_queries.txt --top-k 5
    python scripts/compare_rag_backends.py --user-id 2 --queries-file eval_queries.txt --format jsonl --output rag_compare.jsonl

The script runs each query against python_bm25 and qdrant_hybrid, reports top-k
hits, latency, matched terms, scores, parent context availability, and fallback
warnings. It does not ingest, backfill, delete, or mutate PostgreSQL/Qdrant.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.web_app.core.config import settings
from src.web_app.db.session import SessionLocal
from src.web_app.services.rag_service import rag_service

BACKENDS = ("python_bm25", "qdrant_hybrid")


def main() -> int:
    args = parse_args()
    queries = load_queries(args)
    if not queries:
        print("No queries provided. Use --query or --queries-file.", file=sys.stderr)
        return 2

    document_ids = parse_document_ids(args.document_ids)
    records: list[dict[str, Any]] = []

    with SessionLocal() as db:
        for query in queries:
            for backend in BACKENDS:
                record = run_one(
                    db=db,
                    user_id=args.user_id,
                    query=query,
                    backend=backend,
                    top_k=args.top_k,
                    min_score=args.min_score,
                    document_ids=document_ids,
                )
                records.append(record)

    text = render_jsonl(records) if args.format == "jsonl" else render_markdown(records)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote comparison report: {args.output}")
    else:
        print(text)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run compare python_bm25 vs qdrant_hybrid retrieval.")
    parser.add_argument("--user-id", type=int, required=True, help="User id used for RAG retrieval filters.")
    parser.add_argument("--query", action="append", default=[], help="Query to evaluate. Can be passed multiple times.")
    parser.add_argument("--queries-file", help="UTF-8 text file, one query per line. Blank lines and # comments are ignored.")
    parser.add_argument("--document-ids", help="Optional comma-separated document ids to scope retrieval.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--format", choices=("markdown", "jsonl"), default="markdown")
    parser.add_argument("--output", help="Optional output file path.")
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> list[str]:
    queries = [item.strip() for item in args.query if item and item.strip()]
    if args.queries_file:
        path = Path(args.queries_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                queries.append(item)
    return dedupe(queries)


def run_one(db, user_id: int, query: str, backend: str, top_k: int, min_score: float, document_ids: list[int] | None) -> dict[str, Any]:
    with temporary_backend(backend):
        start = time.perf_counter()
        try:
            response = rag_service.search(
                user_id=user_id,
                query=query,
                top_k=top_k,
                min_score=min_score,
                document_ids=document_ids,
                db=db,
            )
            error = ""
        except Exception as exc:
            response = {"query": query, "results": [], "error": str(exc)}
            error = str(exc)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

    results = [summarize_result(item, rank) for rank, item in enumerate(response.get("results", []), 1)]
    return {
        "query": query,
        "backend": backend,
        "latency_ms": latency_ms,
        "result_count": len(results),
        "warning": response.get("retrieval_warning") or response.get("warning") or response.get("error") or "",
        "error": error,
        "results": results,
    }


def summarize_result(item: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "document_id": item.get("document_id"),
        "filename": item.get("filename") or item.get("source_title", ""),
        "chunk_id": item.get("chunk_id", ""),
        "parent_id": item.get("parent_id"),
        "retrieval_source": item.get("retrieval_source", ""),
        "score": item.get("score"),
        "final_score": item.get("final_score"),
        "vector_score": item.get("vector_score"),
        "bm25_score": item.get("bm25_score"),
        "sparse_score": item.get("sparse_score"),
        "matched_terms": item.get("matched_terms", []),
        "query_type": item.get("query_type", ""),
        "parent_context_available": item.get("parent_context_available", False),
        "heading_path": item.get("heading_path") or (item.get("metadata", {}) or {}).get("heading_path", []),
        "page_number": item.get("page_number") or (item.get("metadata", {}) or {}).get("page_number"),
        "sheet_name": item.get("sheet_name") or (item.get("metadata", {}) or {}).get("sheet_name"),
        "preview": preview(item.get("content") or item.get("content_preview") or ""),
    }


@contextmanager
def temporary_backend(backend: str):
    old_backend = settings.rag_hybrid_backend
    old_fallback = settings.qdrant_hybrid_fallback
    settings.rag_hybrid_backend = backend
    settings.qdrant_hybrid_fallback = True
    try:
        yield
    finally:
        settings.rag_hybrid_backend = old_backend
        settings.qdrant_hybrid_fallback = old_fallback


def render_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# RAG Backend Compare",
        "",
        f"- configured_qdrant_collection: `{settings.qdrant_collection}`",
        f"- qdrant_hybrid_collection: `{settings.qdrant_hybrid_collection}`",
        f"- fallback_enabled: `{settings.qdrant_hybrid_fallback}`",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["query"], []).append(record)

    for query, items in grouped.items():
        lines.extend([f"## Query: {query}", ""])
        for record in items:
            warning = f" warning=`{record['warning']}`" if record.get("warning") else ""
            lines.append(f"### Backend: `{record['backend']}` latency={record['latency_ms']}ms results={record['result_count']}{warning}")
            if not record["results"]:
                lines.append("")
                lines.append("_No results._")
                lines.append("")
                continue
            lines.append("")
            lines.append("| rank | doc | file | chunk | source | score | final | matched_terms | parent | preview |")
            lines.append("|---:|---:|---|---|---|---:|---:|---|---|---|")
            for item in record["results"]:
                lines.append(
                    "| {rank} | {doc} | {file} | {chunk} | {source} | {score} | {final} | {terms} | {parent} | {preview} |".format(
                        rank=item["rank"],
                        doc=item.get("document_id") or "",
                        file=escape_md(str(item.get("filename") or "")),
                        chunk=escape_md(str(item.get("chunk_id") or "")),
                        source=escape_md(str(item.get("retrieval_source") or "")),
                        score=format_score(item.get("score")),
                        final=format_score(item.get("final_score")),
                        terms=escape_md(", ".join(str(term) for term in item.get("matched_terms") or [])),
                        parent="yes" if item.get("parent_context_available") else "no",
                        preview=escape_md(item.get("preview") or ""),
                    )
                )
            lines.append("")
    return "\n".join(lines)


def render_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"


def parse_document_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return ids or None


def preview(text: str, limit: int = 160) -> str:
    compact = " ".join((text or "").split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def format_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
