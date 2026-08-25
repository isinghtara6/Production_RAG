"""
Reference client showing how to call the API correctly, including the
integrity features: checksum, idempotency key, and (optionally) request
signing. Run the server first: `uvicorn app.main:app --reload`.

    python examples/client_example.py
"""
from __future__ import annotations

import hashlib
import time
import uuid

import httpx

BASE_URL = "http://localhost:8000"
KEY_ID = "demo"
SECRET = "password"  # matches the demo hash in .env.example — replace in real use


def auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {KEY_ID}:{SECRET}"}


def signed_headers(body: bytes) -> dict[str, str]:
    """Only needed if REQUIRE_REQUEST_SIGNING=true on the server."""
    ts = str(int(time.time()))
    mac = hashlib.sha256()  # placeholder; real signing uses HMAC, see app/core/security.py
    import hmac as hmac_mod

    sig = hmac_mod.new(SECRET.encode(), (ts + "." ).encode() + body, hashlib.sha256).hexdigest()
    return {"X-Timestamp": ts, "X-Signature": sig}


def main() -> None:
    content = "Cats are small domesticated carnivorous mammals valued by humans for companionship."
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        # 1. Ingest with a client-computed checksum and an idempotency key so
        #    a retried request (e.g. after a network timeout) can't double-ingest.
        ingest_resp = client.post(
            "/v1/documents",
            headers={**auth_header(), "Idempotency-Key": str(uuid.uuid4())},
            json={
                "title": "About Cats",
                "content": content,
                "content_sha256": checksum,
                "source": "example-script",
                "metadata": {"category": "animals"},
            },
        )
        print("ingest:", ingest_resp.status_code, ingest_resp.json())

        # 2. Query
        query_resp = client.post(
            "/v1/query",
            headers=auth_header(),
            json={"query": "What are cats?", "top_k": 3},
        )
        print("query:", query_resp.status_code, query_resp.json())

        # 3. List documents
        list_resp = client.get("/v1/documents", headers=auth_header())
        print("documents:", list_resp.status_code, list_resp.json())


if __name__ == "__main__":
    main()
