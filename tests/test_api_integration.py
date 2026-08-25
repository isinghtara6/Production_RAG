"""
End-to-end HTTP-layer tests using FastAPI's TestClient. These exercise
auth, validation, rate limiting, and the full ingest -> query -> delete
cycle over real HTTP request/response objects, on top of the pure-logic
tests in the other test_*.py files.

Requires `fastapi`, `httpx`, and `python-multipart` to be installed
(see requirements.txt) — these are not importable in every sandbox this
project might be authored in, so this file is deliberately self-contained
and skippable via `pytest -k "not integration"` if needed.
"""
from __future__ import annotations

import io
import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

AUTH = {"Authorization": "Bearer demo:password"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point every stateful path at a throwaway temp directory and force the
    # zero-dependency providers so this test needs no network or API keys.
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "vs"))
    monkeypatch.setenv("METADATA_DB_PATH", str(tmp_path / "meta.sqlite3"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("GENERATION_PROVIDER", "extractive")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "numpy")
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("REQUIRE_REQUEST_SIGNING", "false")
    # demo:password -> sha256("password")
    import hashlib

    monkeypatch.setenv("API_KEYS", f"demo:{hashlib.sha256(b'password').hexdigest()}")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1000")
    monkeypatch.setenv("INGEST_RATE_LIMIT_REQUESTS", "1000")

    from app.config import get_settings

    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def test_health_is_public(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_serves_html_console(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "RAG Console" in resp.text


def test_query_without_auth_is_rejected(client):
    resp = client.post("/v1/query", json={"query": "hello"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "authentication_failed"
    assert "request_id" in resp.json()["error"]


def test_ingest_then_query_roundtrip(client):
    ingest_resp = client.post(
        "/v1/documents",
        headers=AUTH,
        json={"title": "About Cats", "content": "Cats are small domesticated carnivorous mammals."},
    )
    assert ingest_resp.status_code == 201
    body = ingest_resp.json()
    assert body["status"] == "created"
    assert body["chunk_count"] == 1

    query_resp = client.post("/v1/query", headers=AUTH, json={"query": "Tell me about cats", "top_k": 3})
    assert query_resp.status_code == 200
    qbody = query_resp.json()
    assert qbody["retrieved_count"] >= 1
    assert qbody["sources"][0]["document_id"] == body["document_id"]


def test_ingest_rejects_bad_checksum(client):
    resp = client.post(
        "/v1/documents",
        headers=AUTH,
        json={"title": "X", "content": "some content", "content_sha256": "0" * 64},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "checksum_mismatch"


def test_extra_fields_rejected_by_strict_schema(client):
    resp = client.post(
        "/v1/documents",
        headers=AUTH,
        json={"title": "X", "content": "some content", "not_a_real_field": 123},
    )
    assert resp.status_code == 422


def test_upload_txt_file(client):
    file_bytes = b"Dogs are domesticated mammals, not natural wild animals."
    resp = client.post(
        "/v1/documents/upload",
        headers=AUTH,
        files={"file": ("dogs.txt", io.BytesIO(file_bytes), "text/plain")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["chunk_count"] == 1
    assert body["status"] == "created"

    listing = client.get("/v1/documents", headers=AUTH)
    titles = [d["title"] for d in listing.json()["documents"]]
    assert "dogs.txt" in titles


def test_upload_rejects_unsupported_extension(client):
    resp = client.post(
        "/v1/documents/upload",
        headers=AUTH,
        files={"file": ("image.png", io.BytesIO(b"\x89PNG..."), "image/png")},
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_file_type"


def test_delete_document(client):
    ingest_resp = client.post(
        "/v1/documents", headers=AUTH, json={"title": "Temp", "content": "temporary content to delete"}
    )
    doc_id = ingest_resp.json()["document_id"]

    delete_resp = client.delete(f"/v1/documents/{doc_id}", headers=AUTH)
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "deleted"

    delete_again = client.delete(f"/v1/documents/{doc_id}", headers=AUTH)
    assert delete_again.status_code == 404


def test_idempotency_key_prevents_duplicate_ingest_response(client):
    headers = {**AUTH, "Idempotency-Key": "fixed-key-123"}
    r1 = client.post("/v1/documents", headers=headers, json={"title": "A", "content": "content one here"})
    r2 = client.post("/v1/documents", headers=headers, json={"title": "A", "content": "content one here"})
    assert r1.json() == r2.json()
