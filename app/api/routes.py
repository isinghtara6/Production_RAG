"""
Route handlers.

Kept thin on purpose: request validation is Pydantic's job (schemas.py),
auth/signing/rate-limiting are dependencies (deps.py), and all RAG logic
lives in `RagPipeline`. Handlers here are mostly translation between the
HTTP layer and the pipeline, plus audit logging and idempotency bookkeeping.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import (
    enforce_body_size_limit,
    enforce_ingest_rate_limit,
    enforce_query_rate_limit,
    get_pipeline,
)
from app.core.exceptions import DocumentNotFoundError, IdempotencyConflictError, PayloadTooLargeError
from app.core.logging import get_logger, log_extra
from app.core.security import Principal
from app.models.schemas import (
    DeleteResponse,
    DocumentSummary,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ListDocumentsResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from app.rag.file_extraction import extract_text
from app.rag.pipeline import RagPipeline

logger = get_logger(__name__)

router = APIRouter()

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.get("/", include_in_schema=False)
async def serve_test_console() -> FileResponse:
    """Serves the built-in browser test console (app/static/index.html) —
    a zero-build vanilla JS page for exercising the API without curl/Swagger.
    Not intended as a production frontend; disable by removing this route
    (or fronting the API with your own UI) if you don't want it exposed."""
    return FileResponse(_STATIC_DIR / "index.html")


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Liveness probe: process is up. Does not touch dependencies."""
    return HealthResponse(status="ok", service=request.app.state.settings.app_name, version=request.app.state.settings.api_version)


@router.get("/ready", response_model=ReadinessResponse, tags=["ops"])
async def ready(request: Request) -> ReadinessResponse:
    """Readiness probe: can this instance actually serve traffic."""
    pipeline: RagPipeline = request.app.state.pipeline
    checks = {
        "vector_store_loaded": pipeline.vector_store is not None,
        "document_store_reachable": _check_document_store(pipeline),
        "embedder_ready": pipeline.embedder is not None,
    }
    status = "ready" if all(checks.values()) else "not_ready"
    return ReadinessResponse(status=status, checks=checks)


def _check_document_store(pipeline: RagPipeline) -> bool:
    try:
        pipeline.document_store.list(limit=1)
        return True
    except Exception:  # noqa: BLE001
        return False


@router.post(
    "/v1/documents",
    response_model=IngestResponse,
    status_code=201,
    tags=["documents"],
    dependencies=[Depends(enforce_body_size_limit)],
)
async def ingest_document(
    body: IngestRequest,
    request: Request,
    principal: Principal = Depends(enforce_ingest_rate_limit),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> IngestResponse:
    pipeline: RagPipeline = request.app.state.pipeline
    request_id = request.state.request_id

    if idempotency_key:
        cached = pipeline.document_store.get_idempotent_response(idempotency_key)
        if cached is not None:
            return IngestResponse(**cached)

    result = pipeline.ingest(
        document_id=body.document_id,
        title=body.title,
        content=body.content,
        content_sha256_claim=body.content_sha256,
        metadata=body.metadata,
        source=body.source,
    )

    pipeline.document_store.record_audit(
        request_id=request_id, api_key_id=principal.key_id, action=f"ingest:{result['status']}",
        document_id=result["document_id"],
        detail={"chunk_count": result["chunk_count"], "content_sha256": result["content_sha256"]},
    )
    logger.info("document ingested", extra=log_extra(document_id=result["document_id"], status=result["status"]))

    response = IngestResponse(**result)
    if idempotency_key:
        pipeline.document_store.save_idempotent_response(idempotency_key, response.model_dump(mode="json"))

    return response


@router.post(
    "/v1/documents/upload",
    response_model=IngestResponse,
    status_code=201,
    tags=["documents"],
    dependencies=[Depends(enforce_body_size_limit)],
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="A .txt, .md, .pdf, or .docx file."),
    title: str | None = Form(default=None, max_length=500),
    source: str | None = Form(default=None, max_length=1000),
    principal: Principal = Depends(enforce_ingest_rate_limit),
) -> IngestResponse:
    """Accepts a real file (not raw JSON text) and runs it through the same
    checksum-verified, idempotent ingestion path as `POST /v1/documents`.
    Text is extracted server-side based on file extension; see
    app/rag/file_extraction.py for supported formats."""
    pipeline: RagPipeline = request.app.state.pipeline
    settings = request.app.state.settings
    request_id = request.state.request_id

    data = await file.read()
    if len(data) > settings.max_request_body_bytes:
        raise PayloadTooLargeError(
            "Uploaded file exceeds the configured maximum.",
            details={"max_bytes": settings.max_request_body_bytes, "received_bytes": len(data)},
        )

    text = extract_text(file.filename or "upload.txt", data)

    result = pipeline.ingest(
        document_id=None,
        title=title or file.filename or "Untitled",
        content=text,
        content_sha256_claim=None,  # the server computes the checksum directly from the uploaded bytes' extracted text
        metadata={"original_filename": file.filename, "content_type": file.content_type},
        source=source or "file_upload",
    )

    pipeline.document_store.record_audit(
        request_id=request_id, api_key_id=principal.key_id, action=f"upload:{result['status']}",
        document_id=result["document_id"],
        detail={"filename": file.filename, "chunk_count": result["chunk_count"], "bytes": len(data)},
    )
    logger.info(
        "document uploaded",
        extra=log_extra(document_id=result["document_id"], filename=file.filename, status=result["status"]),
    )
    return IngestResponse(**result)


@router.get("/v1/documents", response_model=ListDocumentsResponse, tags=["documents"])
async def list_documents(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    principal: Principal = Depends(enforce_query_rate_limit),
) -> ListDocumentsResponse:
    pipeline: RagPipeline = request.app.state.pipeline
    records, total = pipeline.document_store.list(limit=limit, offset=offset)
    return ListDocumentsResponse(
        documents=[
            DocumentSummary(
                document_id=r.document_id, title=r.title, source=r.source, chunk_count=r.chunk_count,
                content_sha256=r.content_sha256, ingested_at=r.ingested_at, metadata=r.metadata,
            )
            for r in records
        ],
        total=total,
    )


@router.delete("/v1/documents/{document_id}", response_model=DeleteResponse, tags=["documents"])
async def delete_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(enforce_ingest_rate_limit),
) -> DeleteResponse:
    pipeline: RagPipeline = request.app.state.pipeline
    pipeline.delete_document(document_id)
    pipeline.document_store.record_audit(
        request_id=request.state.request_id, api_key_id=principal.key_id,
        action="delete", document_id=document_id, detail={},
    )
    logger.info("document deleted", extra=log_extra(document_id=document_id))
    return DeleteResponse(document_id=document_id, status="deleted")


@router.post(
    "/v1/query",
    response_model=QueryResponse,
    tags=["query"],
    dependencies=[Depends(enforce_body_size_limit)],
)
async def query(
    body: QueryRequest,
    request: Request,
    principal: Principal = Depends(enforce_query_rate_limit),
) -> QueryResponse:
    pipeline: RagPipeline = request.app.state.pipeline
    result = pipeline.query(
        query=body.query, top_k=body.top_k, min_score=body.min_score, request_id=request.state.request_id,
    )
    if not body.include_sources:
        result["sources"] = []

    pipeline.document_store.record_audit(
        request_id=request.state.request_id, api_key_id=principal.key_id, action="query",
        document_id=None, detail={"query_len": len(body.query), "retrieved_count": result["retrieved_count"]},
    )
    return QueryResponse(**result)
