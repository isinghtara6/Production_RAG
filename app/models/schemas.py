"""
Wire-format contracts.

Using `extra="forbid"` everywhere is a deliberate integrity choice: a typo'd
field name from a client fails loudly at validation time instead of being
silently ignored, which is how subtle client/server drift usually starts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------- Ingestion


class IngestRequest(StrictModel):
    document_id: Optional[str] = Field(
        default=None,
        description="Caller-supplied stable ID. If omitted, one is derived from a content hash.",
        max_length=200,
    )
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=2_000_000)
    content_sha256: Optional[str] = Field(
        default=None,
        description="Optional client-computed checksum. If provided, the server "
        "verifies it against the received bytes and rejects a mismatch — "
        "this catches transport-layer corruption or truncation.",
        min_length=64,
        max_length=64,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("content_sha256")
    @classmethod
    def _lowercase_hex(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("content_sha256 must be lowercase hex")
        return v


class ChunkRef(StrictModel):
    chunk_id: str
    index: int
    token_count: int
    checksum: str


class IngestResponse(StrictModel):
    document_id: str
    content_sha256: str
    status: Literal["created", "already_exists", "updated"]
    chunk_count: int
    chunks: list[ChunkRef]
    ingested_at: datetime


class DocumentSummary(StrictModel):
    document_id: str
    title: str
    source: Optional[str]
    chunk_count: int
    content_sha256: str
    ingested_at: datetime
    metadata: dict[str, Any]


class ListDocumentsResponse(StrictModel):
    documents: list[DocumentSummary]
    total: int


class DeleteResponse(StrictModel):
    document_id: str
    status: Literal["deleted"]


# -------------------------------------------------------------------- Query


class QueryRequest(StrictModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    include_sources: bool = True


class RetrievedChunk(StrictModel):
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    score: float
    index: int


class QueryResponse(StrictModel):
    query: str
    answer: str
    sources: list[RetrievedChunk]
    generation_provider: str
    retrieved_count: int
    latency_ms: float
    request_id: str


# ------------------------------------------------------------------- Health


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(StrictModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


class ErrorDetail(StrictModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorDetail
