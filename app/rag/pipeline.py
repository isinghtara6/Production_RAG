"""
Pipeline orchestration.

Wires chunking -> embedding -> vector store -> (metadata store) together for
ingestion, and retrieval -> generation together for querying. This is the
single place that understands the *sequence* of RAG operations; each
component it calls stays independently testable.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from app.core.exceptions import ChecksumMismatchError
from app.core.logging import get_logger
from app.core.security import sha256_hex
from app.rag.chunking import chunk_text
from app.rag.document_store import DocumentRecord, DocumentStore
from app.rag.embeddings import EmbeddingProvider
from app.rag.generator import GenerationProvider
from app.rag.vector_store import VectorRecord, VectorStore

logger = get_logger(__name__)


class RagPipeline:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        document_store: DocumentStore,
        generator: GenerationProvider,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        default_top_k: int,
        min_relevance_score: float,
        vector_store_path: str,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.document_store = document_store
        self.generator = generator
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.default_top_k = default_top_k
        self.min_relevance_score = min_relevance_score
        self.vector_store_path = vector_store_path

    # -------------------------------------------------------------- ingest

    def ingest(
        self,
        *,
        document_id: str | None,
        title: str,
        content: str,
        content_sha256_claim: str | None,
        metadata: dict,
        source: str | None,
    ) -> dict:
        raw = content.encode("utf-8")
        actual_checksum = sha256_hex(raw)

        if content_sha256_claim and content_sha256_claim != actual_checksum:
            raise ChecksumMismatchError(
                "Provided content_sha256 does not match the received content. "
                "The payload may have been corrupted or truncated in transit.",
                details={"expected": content_sha256_claim, "actual": actual_checksum},
            )

        # Idempotent-by-content: re-ingesting byte-identical content is a
        # no-op that returns the existing document rather than duplicating it.
        existing = self.document_store.get_by_checksum(actual_checksum)
        if existing is not None:
            return {
                "document_id": existing.document_id,
                "content_sha256": existing.content_sha256,
                "status": "already_exists",
                "chunk_count": existing.chunk_count,
                "chunks": [],
                "ingested_at": existing.ingested_at,
            }

        doc_id = document_id or f"doc_{actual_checksum[:16]}"
        is_update = self.document_store.get(doc_id) is not None
        if is_update:
            self.vector_store.delete_by_document(doc_id)

        chunks = chunk_text(
            content, chunk_size_tokens=self.chunk_size_tokens, overlap_tokens=self.chunk_overlap_tokens
        )
        if not chunks:
            raise ChecksumMismatchError("Content produced zero chunks after normalization.")

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)

        records = []
        chunk_refs = []
        for c in chunks:
            chunk_id = f"{doc_id}::{c.index}"
            checksum = sha256_hex(c.text.encode("utf-8"))
            records.append(
                VectorRecord(
                    chunk_id=chunk_id, document_id=doc_id, text=c.text, index=c.index,
                    metadata={**metadata, "title": title, "source": source},
                )
            )
            chunk_refs.append(
                {"chunk_id": chunk_id, "index": c.index, "token_count": c.token_count, "checksum": checksum}
            )

        self.vector_store.add(vectors, records)
        self.vector_store.save(self.vector_store_path)

        ingested_at = datetime.now(timezone.utc).isoformat()
        self.document_store.upsert_document(
            DocumentRecord(
                document_id=doc_id, title=title, source=source, content_sha256=actual_checksum,
                chunk_count=len(chunks), metadata=metadata, ingested_at=ingested_at,
            )
        )

        return {
            "document_id": doc_id,
            "content_sha256": actual_checksum,
            "status": "updated" if is_update else "created",
            "chunk_count": len(chunks),
            "chunks": chunk_refs,
            "ingested_at": ingested_at,
        }

    def delete_document(self, document_id: str) -> None:
        self.document_store.delete_document(document_id)
        self.vector_store.delete_by_document(document_id)
        self.vector_store.save(self.vector_store_path)

    # --------------------------------------------------------------- query

    def query(self, *, query: str, top_k: int | None, min_score: float | None, request_id: str) -> dict:
        start = time.perf_counter()
        k = top_k or self.default_top_k
        threshold = self.min_relevance_score if min_score is None else min_score

        query_vec = self.embedder.embed_one(query)
        raw_hits = self.vector_store.search(query_vec, top_k=k)
        hits = [(rec, score) for rec, score in raw_hits if score >= threshold]

        answer = self.generator.generate(query, hits)

        sources = [
            {
                "chunk_id": rec.chunk_id,
                "document_id": rec.document_id,
                "document_title": rec.metadata.get("title", ""),
                "text": rec.text,
                "score": round(score, 6),
                "index": rec.index,
            }
            for rec, score in hits
        ]

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "generation_provider": self.generator.name,
            "retrieved_count": len(sources),
            "latency_ms": round(latency_ms, 2),
            "request_id": request_id,
        }
