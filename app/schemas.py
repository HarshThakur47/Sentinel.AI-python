"""
Pydantic schemas — these define and VALIDATE the API contract.
This is the direct replacement for the manual `validateQuery()` checks
in the Node queryController.js.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    user_query: str = Field(..., alias="userQuery", min_length=1, max_length=1000)

    @field_validator("user_query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty.")
        return trimmed

    model_config = {"populate_by_name": True}


class SourceDoc(BaseModel):
    id: str
    title: str


class PipelineMetadata(BaseModel):
    duration_ms: int = Field(..., alias="durationMs")
    filter_summary: str = Field(..., alias="filterSummary")
    removed_count: int = Field(..., alias="removedCount")
    evaluation_summary: str = Field(..., alias="evaluationSummary")
    is_accurate: bool = Field(..., alias="isAccurate")
    refinement_needed: bool = Field(..., alias="refinementNeeded")
    raw_doc_count: int = Field(..., alias="rawDocCount")
    filtered_doc_count: int = Field(..., alias="filteredDocCount")
    used_general_knowledge: bool = Field(..., alias="usedGeneralKnowledge")

    model_config = {"populate_by_name": True}


class QueryResponse(BaseModel):
    answer: str
    confidence_score: int = Field(..., alias="confidenceScore")
    sources: List[SourceDoc]
    status: str
    metadata: PipelineMetadata

    model_config = {"populate_by_name": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str
