from src.web_app.core.config import settings


def get_qdrant_config() -> dict[str, str | bool]:
    return {"configured": bool(settings.qdrant_url), "url": settings.qdrant_url, "collection": settings.qdrant_collection}


def check_qdrant_health() -> dict[str, object]:
    if not settings.qdrant_url:
        return {"configured": False, "available": False, "message": "Qdrant is not configured"}
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=settings.qdrant_timeout)
        client.get_collections()
        return {"configured": True, "available": True, "message": "Qdrant is available", "collection": settings.qdrant_collection}
    except Exception as exc:
        return {"configured": True, "available": False, "message": str(exc), "collection": settings.qdrant_collection}
