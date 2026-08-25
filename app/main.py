from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import get_settings
from app.core.exceptions import RagServiceError, ValidationFailedError, error_body
from app.core.logging import configure_logging, get_logger
from app.middleware.rate_limit import RateLimiter
from app.middleware.request_id import RequestContextMiddleware
from app.rag.document_store import DocumentStore
from app.rag.embeddings import build_embedder
from app.rag.generator import build_generator
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import VectorStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info("starting up", extra={"extra_fields": {"environment": settings.environment}})

    embedder = build_embedder(settings.embedding_provider, settings.embedding_model, settings.embedding_dim)
    vector_store = VectorStore.load(settings.vector_store_path, dim=embedder.dim, backend=settings.vector_store_backend)
    document_store = DocumentStore(settings.metadata_db_path)
    generator = build_generator(
        settings.generation_provider,
        model=settings.generation_model,
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
        gemini_api_key=settings.gemini_api_key,
    )

    pipeline = RagPipeline(
        embedder=embedder,
        vector_store=vector_store,
        document_store=document_store,
        generator=generator,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        default_top_k=settings.top_k,
        min_relevance_score=settings.min_relevance_score,
        vector_store_path=settings.vector_store_path,
    )

    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.query_rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
    app.state.ingest_rate_limiter = RateLimiter(settings.ingest_rate_limit_requests, settings.ingest_rate_limit_window_seconds)
    app.state.seen_nonces = set()  # replay-protection cache; swap for Redis with TTL in multi-instance deployments

    logger.info(
        "startup complete",
        extra={"extra_fields": {"vectors_loaded": len(vector_store), "embedding_provider": settings.embedding_provider}},
    )
    yield

    vector_store.save(settings.vector_store_path)
    logger.info("shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Timestamp", "X-Signature", "Idempotency-Key", "X-Request-ID"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)
    app.include_router(router)
    return app


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagServiceError)
    async def handle_service_error(request: Request, exc: RagServiceError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        return JSONResponse(status_code=exc.status_code, content=error_body(exc, request_id))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        wrapped = ValidationFailedError("Request failed schema validation.", details={"errors": exc.errors()})
        return JSONResponse(status_code=wrapped.status_code, content=error_body(wrapped, request_id))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        wrapped = RagServiceError(str(exc.detail))
        wrapped.code = "http_error"
        wrapped.status_code = exc.status_code
        return JSONResponse(status_code=exc.status_code, content=error_body(wrapped, request_id))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unhandled exception reached top-level handler")
        wrapped = RagServiceError("An unexpected error occurred.")
        return JSONResponse(status_code=500, content=error_body(wrapped, request_id))


app = create_app()
