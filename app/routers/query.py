"""
Query Router — equivalent of backend/controllers/queryController.js +
backend/routes/queryRoutes.js.

Key differences from the Node version (intentional, showcase-worthy):
  - Request validation is DECLARATIVE via Pydantic (QueryRequest) instead of
    the manual `validateQuery()` function.
  - The route is JWT-protected via `Depends(get_current_user)`.
  - FastAPI auto-generates OpenAPI/Swagger docs for this route — visit
    /docs after starting the server.
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.chains.rag_pipeline import run_rag_pipeline
from app.schemas import QueryRequest, QueryResponse
from app.security import get_current_user

router = APIRouter(prefix="/api/v1", tags=["query"])

_start_time = time.time()


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Sentinel.AI Backend (LangChain Edition)",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": f"{int(time.time() - _start_time)}s",
    }


@router.post("/query", response_model=QueryResponse, response_model_by_alias=True)
async def handle_query(payload: QueryRequest, current_user: str = Depends(get_current_user)):
    """
    POST /api/v1/query
    Body:  { "userQuery": "..." }
    Auth:  Bearer <JWT>   (obtain via POST /api/v1/auth/token)
    """
    result = await run_rag_pipeline(payload.user_query)
    return result
