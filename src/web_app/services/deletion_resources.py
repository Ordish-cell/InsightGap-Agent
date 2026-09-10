"""Strict external cleanup: absence succeeds; unavailable services fail for retry."""
from pathlib import Path
from src.web_app.core.config import settings


def safe_file(path, kind):
    candidate = Path(path).resolve()
    root = Path("storage/uploads" if kind == "document" else settings.artifact_storage_path).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("file_outside_owned_storage")
    if candidate.exists() and not candidate.is_file():
        raise ValueError("expected_owned_file")
    return candidate


def cleanup_resources(user_id, conversation_id, manifest):
    paths = [safe_file(item["path"], item["kind"]) for item in manifest["paths"]]
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    if manifest["document_ids"] or manifest["memory_ids"] or manifest["segment_ids"]:
        if not settings.qdrant_url:
            raise RuntimeError("qdrant_not_configured")
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=settings.qdrant_timeout)
        try:
            names = {c.name for c in client.get_collections().collections}
            scopes = []
            for collection in {settings.qdrant_collection, settings.qdrant_hybrid_collection}:
                scopes.extend((collection, "document_id", i) for i in manifest["document_ids"])
            scopes.extend((settings.memory_qdrant_collection, "memory_id", i) for i in manifest["memory_ids"])
            if manifest["segment_ids"]:
                scopes.append((settings.conversation_segment_vector_collection, "conversation_id", conversation_id))
            def matches(key, value):
                values = [value] if not isinstance(value, int) else [value, str(value)]
                return Filter(should=[FieldCondition(key=key, match=MatchValue(value=v)) for v in values])
            for collection, key, value in scopes:
                if collection not in names:
                    continue
                scope = Filter(must=[matches("user_id", user_id), matches(key, value)])
                client.delete(collection_name=collection, points_selector=scope, wait=True)
                if client.count(collection_name=collection, count_filter=scope, exact=True).count:
                    raise RuntimeError("vector_cleanup_incomplete")
        finally:
            client.close()
    if settings.agent_langgraph_checkpointer_enabled and manifest["checkpoint_threads"]:
        import psycopg
        from src.web_app.agent.runtime.checkpoint_cleanup import _pg_conn_string, CHECKPOINT_DATA_TABLES
        from psycopg import sql
        with psycopg.connect(_pg_conn_string()) as conn:
            with conn.cursor() as cursor:
                for table in CHECKPOINT_DATA_TABLES:
                    cursor.execute("SELECT to_regclass(%s)", (table,))
                    if cursor.fetchone()[0] is None:
                        continue
                    cursor.execute(sql.SQL("DELETE FROM {} WHERE thread_id = ANY(%s)").format(sql.Identifier(table)),
                        (manifest["checkpoint_threads"],))
    if settings.enable_neo4j and settings.neo4j_memory_graph_enabled and manifest["memory_ids"]:
        from src.web_app.graph.neo4j_client import neo4j_client
        # Remove only the projections of selected temporary memories. Shared concepts remain.
        neo4j_client.run_write(
            "MATCH (m:UserMemory {user_id: $user_id}) WHERE m.memory_id IN $memory_ids DETACH DELETE m",
            user_id=user_id, memory_ids=[str(i) for i in manifest["memory_ids"]],
        )
    for path in paths:
        path.unlink(missing_ok=True)
