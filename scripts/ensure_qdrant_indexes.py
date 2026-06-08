"""Ensure Qdrant payload indexes exist for the current collections.

Run this once after deploying or after upgrading Qdrant client.
It is idempotent — safe to run repeatedly.

Usage:
    python scripts/ensure_qdrant_indexes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from src.web_app.core.config import settings


def _index_field(client: QdrantClient, collection: str, field: str) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"  [OK] {collection}.{field} — index created")
    except Exception as exc:
        err_msg = str(exc).lower()
        if "already exists" in err_msg or "already" in err_msg:
            print(f"  [SKIP] {collection}.{field} — already exists")
        else:
            print(f"  [FAIL] {collection}.{field} — {exc}")


def ensure_document_indexes() -> None:
    if not settings.qdrant_url:
        print("QDRANT_URL not set — skipping document indexes")
        return
    collection = settings.qdrant_collection
    print(f"\nEnsuring document indexes on collection: {collection}")
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
    )
    for field in ("user_id", "document_id"):
        _index_field(client, collection, field)


def ensure_memory_indexes() -> None:
    if not settings.qdrant_url:
        print("QDRANT_URL not set — skipping memory indexes")
        return
    collection = settings.memory_qdrant_collection
    print(f"\nEnsuring memory indexes on collection: {collection}")
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
    )
    for field in ("user_id", "memory_id", "memory_type"):
        _index_field(client, collection, field)


if __name__ == "__main__":
    print("Qdrant Payload Index Initialization")
    print("=" * 40)
    print(f"  QDRANT_URL: {settings.qdrant_url or '(not set)'}")
    print(f"  Document collection: {settings.qdrant_collection}")
    print(f"  Memory collection: {settings.memory_qdrant_collection}")
    ensure_document_indexes()
    ensure_memory_indexes()
    print("\nDone.")
