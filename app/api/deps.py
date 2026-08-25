"""
FastAPI dependency wiring.

Auth scheme: `Authorization: Bearer <key_id>:<secret>`. On top of that,
callers may opt into (or be required to send, via REQUIRE_REQUEST_SIGNING)
request signing: `X-Timestamp` (unix seconds) + `X-Signature`
(HMAC-SHA256 of `timestamp.raw_body` using the same secret). This defends
against a stolen API key that's captured in transit-adjacent logs (the key
alone can't forge a valid, fresh signature) and against replay of a
previously captured, correctly-signed request.
"""
from __future__ import annotations

from fastapi import Depends, Header, Request

from app.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, PayloadTooLargeError
from app.core.logging import api_key_id_ctx
from app.core.security import Principal, verify_api_key, verify_signature
from app.middleware.rate_limit import RateLimiter
from app.rag.pipeline import RagPipeline


def get_pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


def get_query_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.query_rate_limiter


def get_ingest_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.ingest_rate_limiter


async def get_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    x_timestamp: str | None = Header(default=None),
    x_signature: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError("Missing or malformed Authorization header. Expected: Bearer <key_id>:<secret>")

    token = authorization[7:]
    if ":" not in token:
        raise AuthenticationError("Malformed bearer token. Expected format: <key_id>:<secret>")
    key_id, secret = token.split(":", 1)

    principal = verify_api_key(key_id, secret, settings.api_key_map)
    api_key_id_ctx.set(principal.key_id)

    if settings.require_request_signing:
        raw_body = await request.body()
        verify_signature(
            secret=secret,
            timestamp=x_timestamp or "",
            signature=x_signature or "",
            raw_body=raw_body,
            tolerance_seconds=settings.signature_tolerance_seconds,
            seen_nonces=request.app.state.seen_nonces,
        )

    return principal


async def enforce_body_size_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > settings.max_request_body_bytes:
        raise PayloadTooLargeError(
            "Request body exceeds the configured maximum.",
            details={"max_bytes": settings.max_request_body_bytes},
        )


def enforce_query_rate_limit(
    principal: Principal = Depends(get_principal),
    limiter: RateLimiter = Depends(get_query_rate_limiter),
) -> Principal:
    limiter.check(principal.key_id)
    return principal


def enforce_ingest_rate_limit(
    principal: Principal = Depends(get_principal),
    limiter: RateLimiter = Depends(get_ingest_rate_limiter),
) -> Principal:
    limiter.check(principal.key_id)
    return principal
