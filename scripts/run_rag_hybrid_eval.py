"""Run a synthetic RAG backend evaluation.

This script creates/uses a dedicated synthetic test user, ingests fixture
documents through DocumentService, compares python_bm25 vs qdrant_hybrid, and
writes Markdown/JSONL reports. It does not use production user documents and it
does not touch memory_vectors.

Default mode is safe: no cleanup, no production deletion, no backfill.
Use --cleanup only to delete documents created by this eval marker.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_rag_backends import render_jsonl, render_markdown, run_one
from src.web_app.core.config import settings
from src.web_app.db.repositories.document_repository import DocumentRepository
from src.web_app.db.session import SessionLocal
from src.web_app.models.orm import Document, DocumentChunk, User
from src.web_app.services.document_service import document_service

FIXTURE_DIR = Path("src/web_app/tests/fixtures/rag_docs")
QUERY_FILE = Path("src/web_app/tests/fixtures/rag_eval_queries.txt")
EXPECTED_FILE = Path("src/web_app/tests/fixtures/rag_eval_expected.json")
DEFAULT_OUTPUT_DIR = Path("uploads/artifacts/rag_eval")
TEST_EMAIL = "rag-eval-synthetic@example.com"
EVAL_MARKER = "synthetic_rag_hybrid_eval"
BACKENDS = ("python_bm25", "qdrant_hybrid")


@dataclass
class EvalMetrics:
    backend: str
    total: int
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    keyword_hit_rate: float
    fallback_count: int
    warning_count: int
    average_latency_ms: float

    @property
    def usable(self) -> bool:
        return self.total > 0 and self.hit_at_3 / self.total >= 0.60 and self.fallback_count == 0


def main() -> int:
    args = parse_args()
    if args.require_staging and settings.app_env.lower() in {"prod", "production"}:
        print("Refusing to run in production app_env without --allow-production.", file=sys.stderr)
        return 3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"rag_hybrid_eval_{run_id}.md"
    jsonl_path = output_dir / f"rag_hybrid_eval_{run_id}.jsonl"

    with SessionLocal() as db:
        user = get_or_create_test_user(db)
        docs = ingest_fixtures(db, user.id, force_reingest=args.force_reingest)
        validate_ingestion(db, user.id, [doc["id"] for doc in docs])
        queries = load_queries(args.queries_file)
        expected = load_expected(args.expected_file)

        records = []
        for query in queries:
            for backend in BACKENDS:
                records.append(
                    run_one(
                        db=db,
                        user_id=user.id,
                        query=query,
                        backend=backend,
                        top_k=args.top_k,
                        min_score=args.min_score,
                        document_ids=[doc["id"] for doc in docs],
                    )
                )

        metrics = compute_metrics(records, expected, top_k=args.top_k)
        report = render_eval_report(records, metrics, docs)
        report_path.write_text(report, encoding="utf-8")
        if args.jsonl:
            jsonl_path.write_text(render_jsonl(records), encoding="utf-8")

        if args.cleanup:
            cleanup_eval_documents(db, user.id, [doc["id"] for doc in docs])

    print(f"Markdown report: {report_path}")
    if args.jsonl:
        print(f"JSONL report: {jsonl_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic RAG hybrid eval against python_bm25 and qdrant_hybrid.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--queries-file", default=str(QUERY_FILE))
    parser.add_argument("--expected-file", default=str(EXPECTED_FILE))
    parser.add_argument("--jsonl", action="store_true", help="Also write JSONL comparison output.")
    parser.add_argument("--force-reingest", action="store_true", help="Reingest existing synthetic eval documents for the test user.")
    parser.add_argument("--cleanup", action="store_true", help="Delete only synthetic eval documents created for the test user after the run.")
    parser.add_argument("--require-staging", action="store_true", default=True)
    parser.add_argument("--allow-production", action="store_true", help="Allow running when APP_ENV=production. Still only touches the synthetic test user.")
    args = parser.parse_args()
    if args.allow_production:
        args.require_staging = False
    return args


def load_fixture_documents(fixture_dir: Path = FIXTURE_DIR) -> list[Path]:
    return sorted(path for path in fixture_dir.iterdir() if path.is_file() and path.suffix.lower() in {".md", ".txt", ".csv"})


def load_queries(path: str | Path = QUERY_FILE) -> list[str]:
    queries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            queries.append(item)
    return queries


def load_expected(path: str | Path = EXPECTED_FILE) -> dict[str, list[str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["query"]: row.get("keywords", []) for row in rows}


def get_or_create_test_user(db) -> User:
    user = db.execute(select(User).where(User.email == TEST_EMAIL)).scalar_one_or_none()
    if user:
        return user
    user = User(email=TEST_EMAIL, hashed_password="synthetic-rag-eval", nickname="RAG Eval Synthetic", status="active")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def ingest_fixtures(db, user_id: int, *, force_reingest: bool = False) -> list[dict[str, Any]]:
    docs = []
    for path in load_fixture_documents():
        existing = find_existing_eval_document(db, user_id, path.name)
        if existing and not force_reingest:
            docs.append({"id": existing.id, "filename": existing.filename, "status": existing.status})
            continue
        if existing and force_reingest:
            document_service.delete_document(db, user_id, existing.id)
        upload = UploadFile(filename=path.name, file=BytesIO(path.read_bytes()))
        uploaded = document_service.upload_document(db, user_id, upload)
        document = DocumentRepository(db).get_by_id_for_user(user_id, int(uploaded["id"]))
        metadata = dict(document.metadata_json or {})
        metadata.update({"eval_marker": EVAL_MARKER, "fixture_name": path.name})
        DocumentRepository(db).update(document, metadata_json=metadata)
        ingested = document_service.ingest_document(db, user_id, document.id)
        docs.append({"id": document.id, "filename": path.name, "status": ingested["document"]["status"]})
    return docs


def find_existing_eval_document(db, user_id: int, filename: str) -> Document | None:
    stmt = select(Document).where(Document.user_id == user_id, Document.filename == filename)
    for document in db.execute(stmt).scalars():
        metadata = document.metadata_json or {}
        if metadata.get("eval_marker") == EVAL_MARKER:
            return document
    return None


def validate_ingestion(db, user_id: int, document_ids: list[int]) -> None:
    for document_id in document_ids:
        chunks = db.execute(select(DocumentChunk).where(DocumentChunk.user_id == user_id, DocumentChunk.document_id == document_id)).scalars().all()
        roles = {(chunk.metadata_json or {}).get("chunk_role") for chunk in chunks}
        if "overview" not in roles or "parent" not in roles or "child" not in roles:
            raise RuntimeError(f"Document {document_id} did not ingest overview/parent/child chunks")
        child_chunks = [chunk for chunk in chunks if (chunk.metadata_json or {}).get("chunk_role") == "child"]
        if not child_chunks:
            raise RuntimeError(f"Document {document_id} has no child chunks")
        if any(not chunk.qdrant_point_id for chunk in child_chunks):
            raise RuntimeError(f"Document {document_id} has child chunks without qdrant_point_id")
        non_child_vectors = [chunk for chunk in chunks if (chunk.metadata_json or {}).get("chunk_role") != "child" and chunk.qdrant_point_id]
        if non_child_vectors:
            raise RuntimeError(f"Document {document_id} has parent/overview qdrant vectors")


def compute_metrics(records: list[dict[str, Any]], expected: dict[str, list[str]], *, top_k: int) -> list[EvalMetrics]:
    metrics = []
    for backend in BACKENDS:
        backend_records = [record for record in records if record["backend"] == backend]
        total = len(backend_records)
        hit1 = hit3 = hit5 = keyword_hits = 0
        fallback_count = 0
        warning_count = 0
        latency_sum = 0.0
        for record in backend_records:
            keywords = expected.get(record["query"], [])
            if keyword_hit(record, keywords, 1):
                hit1 += 1
            if keyword_hit(record, keywords, 3):
                hit3 += 1
            if keyword_hit(record, keywords, min(5, top_k)):
                hit5 += 1
            if keyword_hit(record, keywords, top_k):
                keyword_hits += 1
            warning = record.get("warning") or ""
            if "fallback" in warning.lower() or "failed" in warning.lower():
                fallback_count += 1
            if warning:
                warning_count += 1
            latency_sum += float(record.get("latency_ms") or 0.0)
        metrics.append(EvalMetrics(
            backend=backend,
            total=total,
            hit_at_1=hit1,
            hit_at_3=hit3,
            hit_at_5=hit5,
            keyword_hit_rate=(keyword_hits / total if total else 0.0),
            fallback_count=fallback_count,
            warning_count=warning_count,
            average_latency_ms=(latency_sum / total if total else 0.0),
        ))
    return metrics


def keyword_hit(record: dict[str, Any], keywords: list[str], top_n: int) -> bool:
    if not keywords:
        return False
    haystack = " ".join((item.get("preview") or "") for item in record.get("results", [])[:top_n]).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def render_eval_report(records: list[dict[str, Any]], metrics: list[EvalMetrics], docs: list[dict[str, Any]]) -> str:
    lines = [
        "# Synthetic RAG Hybrid Eval",
        "",
        "## Documents",
        "",
    ]
    for doc in docs:
        lines.append(f"- `{doc['filename']}` document_id={doc['id']} status={doc['status']}")
    lines.extend(["", "## Metrics", "", "| backend | hit@1 | hit@3 | hit@5 | keyword_hit_rate | fallback | warnings | avg_latency_ms | usable |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for metric in metrics:
        total = metric.total or 1
        lines.append(
            f"| {metric.backend} | {metric.hit_at_1 / total:.2f} | {metric.hit_at_3 / total:.2f} | {metric.hit_at_5 / total:.2f} | {metric.keyword_hit_rate:.2f} | {metric.fallback_count} | {metric.warning_count} | {metric.average_latency_ms:.2f} | {'yes' if metric.usable else 'no'} |"
        )
    qdrant = next((metric for metric in metrics if metric.backend == "qdrant_hybrid"), None)
    python = next((metric for metric in metrics if metric.backend == "python_bm25"), None)
    recommendation = "qdrant_hybrid is not ready as default."
    if qdrant and python and qdrant.usable and qdrant.keyword_hit_rate >= python.keyword_hit_rate and qdrant.average_latency_ms <= python.average_latency_ms * 1.5:
        recommendation = "qdrant_hybrid looks eligible for default in this synthetic eval."
    lines.extend(["", "## Recommendation", "", recommendation, "", render_markdown(records)])
    return "\n".join(lines)


def cleanup_eval_documents(db, user_id: int, document_ids: list[int]) -> None:
    for document_id in document_ids:
        document = DocumentRepository(db).get_by_id_for_user(user_id, int(document_id))
        if not document:
            continue
        if (document.metadata_json or {}).get("eval_marker") != EVAL_MARKER:
            raise RuntimeError(f"Refusing to clean non-eval document {document_id}")
        document_service.delete_document(db, user_id, document.id)


if __name__ == "__main__":
    raise SystemExit(main())
