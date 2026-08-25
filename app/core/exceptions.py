"""
Error taxonomy for the API.

Every error the client can see maps to a stable `code` string, an HTTP
status, and an optional `details` object. This is a contract: clients
should branch on `code`, never on the human-readable `message`, which may
change wording over time. Internal exception details (stack traces, DB
errors) are logged but never leaked into the HTTP response body.
"""
from __future__ import annotations

from typing import Any


class RagServiceError(Exception):
    """Base class for all deliberately-raised, client-facing errors."""

    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(RagServiceError):
    code = "authentication_failed"
    status_code = 401


class AuthorizationError(RagServiceError):
    code = "authorization_failed"
    status_code = 403


class SignatureInvalidError(RagServiceError):
    code = "signature_invalid"
    status_code = 401


class ReplayDetectedError(RagServiceError):
    code = "replay_detected"
    status_code = 401


class ValidationFailedError(RagServiceError):
    code = "validation_failed"
    status_code = 422


class PayloadTooLargeError(RagServiceError):
    code = "payload_too_large"
    status_code = 413


class RateLimitExceededError(RagServiceError):
    code = "rate_limit_exceeded"
    status_code = 429


class DocumentNotFoundError(RagServiceError):
    code = "document_not_found"
    status_code = 404


class ChecksumMismatchError(RagServiceError):
    code = "checksum_mismatch"
    status_code = 409


class IdempotencyConflictError(RagServiceError):
    code = "idempotency_conflict"
    status_code = 409


class GenerationProviderError(RagServiceError):
    code = "generation_provider_error"
    status_code = 502


class VectorStoreError(RagServiceError):
    code = "vector_store_error"
    status_code = 500


def error_body(exc: RagServiceError, request_id: str) -> dict[str, Any]:
    return {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id,
        }
    }
