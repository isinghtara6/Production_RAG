"""
Every request gets a correlation ID: taken from an inbound `X-Request-ID`
header if the caller supplied one (useful when a gateway/client wants to
correlate its own logs with ours), otherwise generated. It's echoed back on
the response and bound into the logging context for the lifetime of the
request, so every log line emitted while handling it is traceable.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, log_extra, request_id_ctx

logger = get_logger("access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("x-request-id")
        request_id = incoming if incoming else f"req_{uuid.uuid4().hex}"
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "unhandled exception",
                extra=log_extra(path=request.url.path, method=request.method, duration_ms=round(duration_ms, 2)),
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "request completed",
                extra=log_extra(
                    path=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2),
                ),
            )
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)
