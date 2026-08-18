from fastapi import APIRouter

from src.web_app.api.v1 import agent, approvals, artifacts, auth, documents, feed, health, llm, mcp, memory, profile, research, skills

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(profile.router, prefix="/profile", tags=["profile"])
api_router.include_router(feed.router, prefix="/feed", tags=["feed"])
api_router.include_router(documents.router, prefix="", tags=["documents", "rag"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(research.router, prefix="/research", tags=["research"])
api_router.include_router(skills.router, prefix="/skills", tags=["skills"])
api_router.include_router(artifacts.router, prefix="/artifacts", tags=["artifacts"])
api_router.include_router(approvals.router, prefix="/approvals", tags=["approvals"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
