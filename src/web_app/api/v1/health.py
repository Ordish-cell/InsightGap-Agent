from fastapi import APIRouter

from src.web_app.core.config import settings
from src.web_app.db.session import check_mysql_health, check_redis_health
from src.web_app.rag.qdrant_client import check_qdrant_health
from src.web_app.rag.embeddings import get_embedding_dimension
from src.web_app.services.feed_service import source_health
from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter
from src.web_app.agent.runtime.graph import AgentRuntime
from src.web_app.services.mcp_service import mcp_service
from src.web_app.schemas.common import ok

router = APIRouter()


@router.get("")
def health():
    return ok({"status": "ok"})


@router.get("/dependencies")
def dependencies():
    return ok(
        {
            "mysql": check_mysql_health(),
            "redis": check_redis_health(),
            "qdrant": check_qdrant_health(),
            "qdrant_collection": settings.qdrant_collection,
            "embedding_provider": settings.embed_model_type,
            "embedding_dimension": get_embedding_dimension(),
            "feed_sources": source_health(),
            "open_deep_research": OpenDeepResearchAdapter().health(),
            "agent_runtime": {"status": "ok", "adapter": "langgraph", "fallback_enabled": True, "available": bool(AgentRuntime)},
            "mcp": mcp_service.health(),
            "neo4j": {
                "enabled": settings.enable_neo4j,
                "configured": bool(settings.neo4j_uri),
                "available": False,
                "message": "Neo4j is reserved for later phase",
            },
        }
    )
