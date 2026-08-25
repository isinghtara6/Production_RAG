"""
Document metadata store.

SQLite is deliberately chosen over "just keep it in memory": it gives us a
durable, ACID record of what's been ingested (survives process restarts),
a natural place to enforce the content-checksum uniqueness constraint that
powers idempotent ingestion, and an audit log table that answers "who
ingested/deleted what, and when" — a baseline requirement for any system
claiming API integrity. For multi-node deployment, point this at Postgres
instead; the SQL here is intentionally portable.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from app.core.exceptions import DocumentNotFoundError


@dataclass
class DocumentRecord:
    document_id: str
    title: str
    source: Optional[str]
    content_sha256: str
    chunk_count: int
    metadata: dict[str, Any]
    ingested_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source          TEXT,
    content_sha256  TEXT NOT NULL,
    chunk_count     INTEGER NOT NULL,
    metadata_json   TEXT NOT NULL,
    ingested_at     TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_checksum ON documents(content_sha256);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    request_id  TEXT NOT NULL,
    api_key_id  TEXT NOT NULL,
    action      TEXT NOT NULL,
    document_id TEXT,
    detail_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    response_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


class DocumentStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------- writes

    def upsert_document(self, rec: DocumentRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO documents
                   (document_id, title, source, content_sha256, chunk_count, metadata_json, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                       title=excluded.title, source=excluded.source,
                       content_sha256=excluded.content_sha256, chunk_count=excluded.chunk_count,
                       metadata_json=excluded.metadata_json, ingested_at=excluded.ingested_at
                """,
                (
                    rec.document_id, rec.title, rec.source, rec.content_sha256,
                    rec.chunk_count, json.dumps(rec.metadata), rec.ingested_at,
                ),
            )

    def delete_document(self, document_id: str) -> None:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            if cur.rowcount == 0:
                raise DocumentNotFoundError(f"Document '{document_id}' does not exist.")

    def get_by_checksum(self, content_sha256: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT document_id, title, source, content_sha256, chunk_count, metadata_json, ingested_at "
                "FROM documents WHERE content_sha256 = ?",
                (content_sha256,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT document_id, title, source, content_sha256, chunk_count, metadata_json, ingested_at "
                "FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self, limit: int = 100, offset: int = 0) -> tuple[list[DocumentRecord], int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            rows = conn.execute(
                "SELECT document_id, title, source, content_sha256, chunk_count, metadata_json, ingested_at "
                "FROM documents ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(r) for r in rows], total

    @staticmethod
    def _row_to_record(row: tuple) -> DocumentRecord:
        return DocumentRecord(
            document_id=row[0], title=row[1], source=row[2], content_sha256=row[3],
            chunk_count=row[4], metadata=json.loads(row[5]), ingested_at=row[6],
        )

    # ----------------------------------------------------------- audit log

    def record_audit(
        self, *, request_id: str, api_key_id: str, action: str,
        document_id: Optional[str], detail: dict[str, Any],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, request_id, api_key_id, action, document_id, detail_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(), request_id, api_key_id,
                    action, document_id, json.dumps(detail),
                ),
            )

    # --------------------------------------------------------- idempotency

    def get_idempotent_response(self, key: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM idempotency_keys WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_idempotent_response(self, key: str, response: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO idempotency_keys (idempotency_key, response_json, created_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(response), datetime.now(timezone.utc).isoformat()),
            )
