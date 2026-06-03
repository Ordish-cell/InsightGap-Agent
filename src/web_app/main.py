from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.web_app.api.v1.router import api_router
from src.web_app.core.config import settings

app = FastAPI(title="Open Deep Research Agent OS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", "")
    return await call_next(request)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "FastAPI is running",
    }


app.include_router(api_router, prefix="/api/v1")
