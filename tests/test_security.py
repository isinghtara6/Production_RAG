import time

import pytest

from app.core.exceptions import AuthenticationError, ReplayDetectedError, SignatureInvalidError
from app.core.security import (
    compute_signature,
    hash_secret,
    sha256_hex,
    verify_api_key,
    verify_signature,
)


def test_verify_api_key_success():
    secret = "correct-horse-battery-staple"
    key_map = {"demo": hash_secret(secret)}
    principal = verify_api_key("demo", secret, key_map)
    assert principal.key_id == "demo"


def test_verify_api_key_unknown_id_rejected():
    with pytest.raises(AuthenticationError):
        verify_api_key("nope", "whatever", {})


def test_verify_api_key_wrong_secret_rejected():
    key_map = {"demo": hash_secret("right-secret")}
    with pytest.raises(AuthenticationError):
        verify_api_key("demo", "wrong-secret", key_map)


def test_signature_roundtrip_succeeds():
    secret = "s3cr3t"
    body = b'{"query": "hello"}'
    ts = str(int(time.time()))
    sig = compute_signature(secret, ts, body)
    # Should not raise
    verify_signature(secret=secret, timestamp=ts, signature=sig, raw_body=body, tolerance_seconds=300)


def test_signature_tampered_body_rejected():
    secret = "s3cr3t"
    ts = str(int(time.time()))
    sig = compute_signature(secret, ts, b'{"query": "hello"}')
    with pytest.raises(SignatureInvalidError):
        verify_signature(
            secret=secret, timestamp=ts, signature=sig,
            raw_body=b'{"query": "hello but modified"}', tolerance_seconds=300,
        )


def test_signature_stale_timestamp_rejected():
    secret = "s3cr3t"
    body = b"payload"
    old_ts = str(int(time.time()) - 10_000)
    sig = compute_signature(secret, old_ts, body)
    with pytest.raises(ReplayDetectedError):
        verify_signature(secret=secret, timestamp=old_ts, signature=sig, raw_body=body, tolerance_seconds=300)


def test_signature_replay_detected_via_nonce_cache():
    secret = "s3cr3t"
    body = b"payload"
    ts = str(int(time.time()))
    sig = compute_signature(secret, ts, body)
    seen: set[str] = set()
    verify_signature(secret=secret, timestamp=ts, signature=sig, raw_body=body, tolerance_seconds=300, seen_nonces=seen)
    with pytest.raises(ReplayDetectedError):
        verify_signature(secret=secret, timestamp=ts, signature=sig, raw_body=body, tolerance_seconds=300, seen_nonces=seen)


def test_sha256_hex_matches_known_vector():
    # sha256("") is a well-known constant test vector.
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert len(sha256_hex(b"hello")) == 64
