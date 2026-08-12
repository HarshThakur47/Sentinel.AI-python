"""
Sentinel.AI (LangChain Edition) — FastAPI Entry Point

Equivalent of backend/app.js + backend/server.js combined. Run with:
    uvicorn app.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.routers import auth, query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("sentinel.main")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Sentinel.AI Backend (LangChain Edition)",
    description="A LangChain + Hugging Face + FastAPI reimplementation of the "
    "original Node.js multi-agent RAG pipeline.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(auth.router)
app.include_router(query.router)


@app.get("/")
async def root():
    return {
        "service": "Sentinel.AI Backend (LangChain Edition)",
        "version": "1.0.0",
        "status": "running",
        "docsHint": "GET /docs for interactive API docs | POST /api/v1/auth/token | POST /api/v1/query",
    }


@app.on_event("startup")
async def startup_event():
    if not settings.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set. LLM calls will fail against real models.")
    logger.info("-" * 55)
    logger.info("  Sentinel.AI Backend (LangChain Edition)")
    logger.info(f"  Server        : http://localhost:{settings.PORT}")
    logger.info(f"  Docs (Swagger): http://localhost:{settings.PORT}/docs")
    logger.info(f"  Query API     : POST http://localhost:{settings.PORT}/api/v1/query")
    logger.info(f"  Auth          : POST http://localhost:{settings.PORT}/api/v1/auth/token")
    logger.info(f"  Health check  : GET  http://localhost:{settings.PORT}/api/v1/health")
    logger.info("-" * 55)
