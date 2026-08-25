# Production RAG Service

A Retrieval-Augmented Generation system exposed over a FastAPI HTTP API,
built around one goal beyond "answers questions over documents": **you can
trust the API**. Every layer that can silently fail or be silently abused —
who's calling, whether the payload survived transit, whether it's been seen
before, how fast someone can hit it — is addressed explicitly rather than
assumed away.

## Architecture

```
Client
  │  Authorization: Bearer <key_id>:<secret>
  │  X-Timestamp / X-Signature (optional, HMAC)
  ▼
FastAPI app  ──► RequestContextMiddleware (correlation ID, access log)
  │             ──► CORS / GZip
  │             ──► exception handlers → uniform {error:{code,message,...}}
  ▼
Dependencies  ──► auth (API key) → signature verify → body-size limit → rate limit
  ▼
Routes (app/api/routes.py) — thin HTTP <-> pipeline translation, audit logging
  ▼
RagPipeline (app/rag/pipeline.py)
  ├── chunking.py      token-windowed splitting with overlap
  ├── embeddings.py     EmbeddingProvider: HashEmbedder (offline) | SentenceTransformerEmbedder
  ├── vector_store.py   VectorStore: numpy brute-force | FAISS, disk-persisted
  ├── document_store.py SQLite: document metadata, checksum index, audit log, idempotency
  └── generator.py      GenerationProvider: ExtractiveGenerator (offline) | Anthropic | OpenAI
```

Every provider (embedding, vector store, generation) sits behind a small
interface so a component can be swapped — a hosted embedding API, a real
vector database, Postgres instead of SQLite — without touching the pipeline
or the HTTP layer.

## API integrity model

| Concern | Mechanism | Where |
|---|---|---|
| **Who is calling** | API key (`key_id:secret`), secret stored server-side only as a SHA-256 hash, timing-safe comparison | `core/security.py::verify_api_key` |
| **Tamper detection** | Optional HMAC-SHA256 request signing over `timestamp.body` | `core/security.py::verify_signature` |
| **Replay protection** | Signature timestamp tolerance window + seen-nonce cache | same |
| **Payload integrity** | Client-supplied `content_sha256`, verified server-side before ingestion | `rag/pipeline.py::ingest` |
| **Duplicate/retry safety** | Content-hash dedupe (byte-identical re-ingest is a no-op) + `Idempotency-Key` header for exact response replay | `document_store.py`, `routes.py` |
| **Input validation** | Pydantic v2 models, `extra="forbid"`, length/range bounds on every field | `models/schemas.py` |
| **Abuse throttling** | Per-API-key token-bucket rate limiting, separate budgets for ingest vs. query | `middleware/rate_limit.py` |
| **Oversized payloads** | `Content-Length` checked against `MAX_REQUEST_BODY_BYTES` before parsing | `api/deps.py::enforce_body_size_limit` |
| **Auditability** | Every ingest/delete/query is logged with request ID, API key ID, and outcome to a durable audit table | `document_store.py::record_audit` |
| **Traceability** | Every request gets a correlation ID, propagated through logs and returned as `X-Request-ID` | `middleware/request_id.py` |
| **Stable error contract** | Every error response is `{"error": {"code", "message", "details", "request_id"}}`; clients branch on `code`, not prose | `core/exceptions.py` |
| **Schema contract enforcement** | `extra="forbid"` rejects unknown fields instead of silently dropping them | `models/schemas.py` |

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Then open **`http://localhost:8000/`** — that's a built-in browser test
console (`app/static/index.html`), not a 404. Use it to upload files or
paste text, run queries, and watch a live "manifest" panel showing the
checksum, request ID, and latency of every call — no curl or Postman
needed. See [Browser test console](#browser-test-console) below.

Swagger UI (interactive OpenAPI docs) is at `http://localhost:8000/docs`
(disabled automatically when `ENVIRONMENT=production`).

### Docker

```bash
docker compose up --build
```

Then visit `http://localhost:8000/` the same way.

## How to test

There are three ways to exercise this API, from easiest to most thorough:

**1. The browser console (`http://localhost:8000/`)** — click around: upload
a `.txt`/`.md`/`.pdf`/`.docx` file or paste text, then ask a question about
it. The right-hand "Request Manifest" panel shows you the raw JSON response
and integrity metadata (checksum match, request ID, latency) for every call,
which doubles as a way to *see* the integrity features working, not just
trust that they exist.

**2. curl / the example script** — see [API surface](#api-surface) below for
exact commands, or run:
```bash
pip install httpx
python examples/client_example.py
```

**3. The automated test suite:**
```bash
pip install -r requirements.txt   # pytest is included
pytest -v
```
| File | What it covers | Needs |
|---|---|---|
| `test_chunking.py`, `test_vector_store.py`, `test_security.py` | Pure logic: token windowing, cosine search, HMAC/checksum verification | Nothing — no network, no API keys |
| `test_file_extraction.py` | .txt/.md/.pdf/.docx text extraction, including real generated fixtures | `pypdf` / `python-docx` for the PDF/DOCX cases (skipped automatically if absent) |
| `test_pipeline.py` | Full ingest → query → delete cycle against the RAG pipeline directly | Nothing |
| `test_api_integration.py` | Same cycle over real HTTP requests (auth, validation, rate limits, upload endpoint, the `/` console) via FastAPI's `TestClient` | `fastapi`, `httpx` (skipped automatically if absent) |

Run just the dependency-free subset with `pytest -k "not integration"` if
you're iterating without the full install.

### How to get output

- **From the console**: the answer and its sources render directly on the
  Query tab after you click "Ask" — no extra steps.
- **From curl**: every endpoint returns JSON directly to stdout; pipe
  through `| python -m json.tool` or `| jq` for readability.
- **From pytest**: run with `-v` for a pass/fail line per test, or add
  `-s` to see any `print()` output (e.g. from `examples/client_example.py`
  if you adapt it into a test).
- **Logs**: the running server prints one structured JSON line per request
  to stdout (via `app/core/logging.py`) — `docker compose logs -f rag-api`
  if running in Docker, or just watch the terminal running `uvicorn`.

## Browser test console

`GET /` serves `app/static/index.html` — a single-file, zero-build vanilla
JS page that talks to the API using `fetch`. It's meant for hands-on
testing, not as a production frontend for end users. What it does:

- **Connection panel**: set the API base URL and your `key_id:secret` (saved
  in `localStorage` only — nothing is sent anywhere but your own API).
- **Ingest → Upload file**: drag-and-drop or pick a `.txt`/`.md`/`.pdf`/`.docx`
  file, posts to `/v1/documents/upload`.
- **Ingest → Paste text**: computes a SHA-256 checksum of your text
  *in the browser* (via `crypto.subtle`) before sending it, so you can watch
  the server-side checksum verification in `rag/pipeline.py::ingest` actually
  match it — the manifest panel shows "match ✓" or a mismatch.
- **Query**: ask a question, adjust `top_k`, see the generated answer with
  its source chunks and relevance scores.
- **Documents sidebar**: lists everything ingested, with one-click delete.
- **Request Manifest rail**: every call leaves a "stamped" card showing its
  request ID, server latency, and checksum/verification status, plus the
  raw JSON response — this is the fastest way to confirm the integrity
  features (auth, checksums, request IDs) are doing what they claim.

If you don't want this exposed (e.g. in a real production deployment),
delete the `serve_test_console` route in `app/api/routes.py` — everything
else keeps working unchanged, since it's just one more route.

## Configuration

All configuration is environment-variable driven — see `.env.example` for
the full, documented list. Highlights:

- `EMBEDDING_PROVIDER=hash` (default, offline) or `sentence_transformers`
  for real semantic embeddings.
- `GENERATION_PROVIDER=extractive` (default, offline) or `anthropic` /
  `openai` / `gemini` for LLM-synthesized answers — set the matching
  `*_API_KEY`. For Gemini: `GENERATION_PROVIDER=gemini`,
  `GEMINI_API_KEY=<your key from https://aistudio.google.com/apikey>`,
  and set `GENERATION_MODEL` to a Gemini model name (e.g.
  `gemini-2.5-flash`) — it defaults to an Anthropic model name otherwise,
  which the Gemini provider will refuse and fall back from automatically.
- `VECTOR_STORE_BACKEND=numpy` (default, exact brute-force) or `faiss` for
  larger corpora.
- `REQUIRE_REQUEST_SIGNING=true` to make HMAC signing mandatory, not just
  supported.
- `API_KEYS=key_id:sha256(secret),...` — generate with:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  python -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" '<secret>'
  ```

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Browser test console (see above) |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/ready` | Readiness probe (checks dependencies) |
| `POST` | `/v1/documents` | Ingest a document from raw JSON text (chunk, embed, index) |
| `POST` | `/v1/documents/upload` | Ingest a document from an uploaded `.txt`/`.md`/`.pdf`/`.docx` file |
| `GET`  | `/v1/documents` | List ingested documents |
| `DELETE` | `/v1/documents/{document_id}` | Remove a document and its vectors |
| `POST` | `/v1/query` | Retrieve relevant chunks and generate an answer |

Every route except `/`, `/health`, and `/docs` requires
`Authorization: Bearer <key_id>:<secret>`. Quick reference:

```bash
# Ingest raw text
curl -X POST http://localhost:8000/v1/documents \
  -H "Authorization: Bearer demo:password" -H "Content-Type: application/json" \
  -d '{"title": "About Cats", "content": "Cats are small domesticated mammals."}'

# Ingest a file
curl -X POST http://localhost:8000/v1/documents/upload \
  -H "Authorization: Bearer demo:password" \
  -F "file=@/path/to/your/document.pdf"

# Ask a question
curl -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer demo:password" -H "Content-Type: application/json" \
  -d '{"query": "What are cats?", "top_k": 3}'
```

`demo:password` matches the demo hash shipped in `.env.example` — replace
it with your own generated key before using this for anything real (see
Configuration above).

See `examples/client_example.py` for a full worked Python example (auth,
checksum, idempotency key, optional signing).

## Scaling beyond this reference implementation

This ships with SQLite + an in-process numpy/FAISS index + in-memory rate
limiter, which is correct and genuinely production-usable for a single
instance. To run multiple replicas behind a load balancer:

1. **Vector store**: point `VectorStore` at a networked vector database
   (pgvector, Qdrant, Pinecone) so replicas share one index instead of each
   holding a private copy.
2. **Metadata store**: swap SQLite for Postgres (the SQL in
   `document_store.py` is written to be portable).
3. **Rate limiting**: swap the in-memory token bucket for Redis so limits
   are enforced across replicas, not per-process.
4. **Replay-nonce cache**: same — move `app.state.seen_nonces` to Redis
   with a TTL matching `SIGNATURE_TOLERANCE_SECONDS`.
5. Put a reverse proxy / API gateway in front for TLS termination.

None of these changes touch the HTTP layer or route handlers — that's the
point of the provider interfaces in `rag/*.py`.

## Deploying somewhere real

The Docker image (`Dockerfile`) is the deployment artifact — any host that
runs a container works. A few concrete, low-effort options:

**Render / Railway / Fly.io (free or cheap tiers, simplest path)**
1. Push this project to a GitHub repo.
2. Create a new service on the platform, point it at the repo — all three
   auto-detect the `Dockerfile`.
3. Set environment variables from `.env.example` in the platform's
   dashboard (at minimum, a real `API_KEYS` value — don't ship the demo
   key). Attach a persistent volume mounted at `/app/data` if the platform
   supports it, so `VECTOR_STORE_PATH`/`METADATA_DB_PATH` survive restarts;
   otherwise the index rebuilds empty on every redeploy.
4. Deploy. Your console is then live at `https://<your-app>.<platform>.app/`.

**Any VM / bare server**
```bash
git clone <your-repo> && cd rag_system
cp .env.example .env   # edit API_KEYS, GENERATION_PROVIDER, etc.
docker compose up -d --build
```
Put nginx or Caddy in front for TLS; both are a few lines of config to
reverse-proxy to `localhost:8000`.

**Before exposing this to the internet, at minimum:**
- Generate a real `API_KEYS` value (see Configuration above) — never ship
  the `demo:password` key from `.env.example`.
- Set `ENVIRONMENT=production` (disables `/docs` and `/redoc`).
- Consider `REQUIRE_REQUEST_SIGNING=true` if clients can pre-share a secret.
- Narrow `ALLOWED_ORIGINS` to your actual frontend's origin instead of the
  default `localhost:3000`.
- Put a real vector DB / Postgres / Redis behind the provider interfaces
  (see "Scaling" above) if you expect more than one instance or a large
  corpus — the SQLite/numpy defaults are single-instance by design.
