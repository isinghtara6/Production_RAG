"""
API integrity primitives.

Three independent layers, each defending against a different failure mode:

1. Authentication (API key)      -> "who is calling us?"
2. Request signing (HMAC)        -> "did the body arrive unmodified, and is
                                      this call fresh (not a replay)?"
3. Content checksums (SHA-256)   -> "did the *document* we stored survive
                                      transit / disk intact, and can we
                                      detect duplicate ingestion?"

Secrets are never compared with `==` (timing side-channel); we use
`hmac.compare_digest` everywhere. API key secrets are stored server-side
only as a SHA-256 hash (see .env.example) so a leaked config file does not
hand over live credentials.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Optional

from app.core.exceptions import AuthenticationError, ReplayDetectedError, SignatureInvalidError


@dataclass(frozen=True)
class Principal:
    key_id: str


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def verify_api_key(key_id: str, secret: str, api_key_map: dict[str, str]) -> Principal:
    """Constant-time verification against the configured key->hash map."""
    expected_hash = api_key_map.get(key_id)
    if expected_hash is None:
        # Do the hash+compare anyway (against a dummy) to avoid a timing
        # oracle that reveals which key_ids exist.
        hmac.compare_digest(hash_secret(secret), hash_secret("dummy"))
        raise AuthenticationError("Unknown API key.")
    if not hmac.compare_digest(hash_secret(secret), expected_hash):
        raise AuthenticationError("Invalid API key.")
    return Principal(key_id=key_id)


def compute_signature(secret: str, timestamp: str, raw_body: bytes) -> str:
    """HMAC-SHA256 over `timestamp.body`, hex-encoded. Binding the timestamp
    into the signature is what makes replay detection possible below."""
    mac = hmac.new(secret.encode("utf-8"), digestmod=hashlib.sha256)
    mac.update(timestamp.encode("utf-8"))
    mac.update(b".")
    mac.update(raw_body)
    return mac.hexdigest()


def verify_signature(
    *,
    secret: str,
    timestamp: str,
    signature: str,
    raw_body: bytes,
    tolerance_seconds: int,
    seen_nonces: Optional[set[str]] = None,
) -> None:
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        raise SignatureInvalidError("Missing or malformed X-Timestamp header.")

    now = int(time.time())
    if abs(now - ts) > tolerance_seconds:
        raise ReplayDetectedError(
            "Request timestamp is outside the allowed tolerance window.",
            details={"tolerance_seconds": tolerance_seconds},
        )

    expected = compute_signature(secret, timestamp, raw_body)
    if not hmac.compare_digest(expected, signature or ""):
        raise SignatureInvalidError("Request signature does not match body.")

    if seen_nonces is not None:
        nonce = f"{timestamp}:{signature}"
        if nonce in seen_nonces:
            raise ReplayDetectedError("This exact signed request was already processed.")
        seen_nonces.add(nonce)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
