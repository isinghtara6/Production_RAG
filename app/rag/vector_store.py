"""
Vector store.

Two backends behind one interface:
  - numpy: exact brute-force cosine search. O(n) per query but zero extra
    dependencies and perfectly accurate — the right default until the
    corpus is large enough that latency actually suffers.
  - faiss: approximate/exact ANN search via FAISS, for corpora where
    brute-force numpy stops being fast enough.

Persistence is deliberately simple (vectors as a .npy, metadata as .json)
so the store's on-disk format is inspectable and diffable, which matters
for an "integrity" story: you can checksum the persisted files and detect
silent corruption between restarts.

All mutating operations take a lock — this store is safe to share across
FastAPI's async worker threads/tasks within a single process. For
multi-process/multi-node deployment, swap this for a real vector DB
(pgvector, Qdrant, Pinecone, ...) behind the same `VectorStore` interface.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.exceptions import VectorStoreError


@dataclass
class VectorRecord:
    chunk_id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self, dim: int, backend: str = "numpy") -> None:
        self.dim = dim
        self.backend = backend
        self._lock = threading.RLock()
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._records: list[VectorRecord] = []
        self._id_to_row: dict[str, int] = {}
        self._faiss_index = None

        if backend == "faiss":
            try:
                import faiss  # type: ignore

                self._faiss = faiss
                self._faiss_index = faiss.IndexFlatIP(dim)  # cosine sim via normalized vectors
            except ImportError as e:
                raise VectorStoreError(
                    "faiss backend requested but faiss-cpu is not installed."
                ) from e

    # ------------------------------------------------------------ mutation

    def add(self, vectors: np.ndarray, records: list[VectorRecord]) -> None:
        if len(records) != vectors.shape[0]:
            raise VectorStoreError("vectors/records length mismatch")
        if vectors.shape[1] != self.dim:
            raise VectorStoreError(f"expected dim {self.dim}, got {vectors.shape[1]}")

        with self._lock:
            start_row = self._vectors.shape[0]
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
            for offset, rec in enumerate(records):
                self._id_to_row[rec.chunk_id] = start_row + offset
                self._records.append(rec)
            if self._faiss_index is not None:
                self._faiss_index.add(vectors.astype(np.float32))

    def delete_by_document(self, document_id: str) -> int:
        """Removes all chunks belonging to a document. Rebuilds the dense
        array (O(n)) — acceptable for the moderate corpus sizes this
        reference store targets; swap backends for very large corpora with
        frequent deletes."""
        with self._lock:
            keep_idx = [i for i, r in enumerate(self._records) if r.document_id != document_id]
            removed = len(self._records) - len(keep_idx)
            if removed == 0:
                return 0
            self._vectors = self._vectors[keep_idx] if keep_idx else np.zeros((0, self.dim), dtype=np.float32)
            self._records = [self._records[i] for i in keep_idx]
            self._id_to_row = {r.chunk_id: i for i, r in enumerate(self._records)}
            if self._faiss_index is not None:
                self._faiss.write_index  # no in-place delete for FlatIP; rebuild
                self._faiss_index = self._faiss.IndexFlatIP(self.dim)
                if self._vectors.shape[0]:
                    self._faiss_index.add(self._vectors)
            return removed

    # -------------------------------------------------------------- search

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[VectorRecord, float]]:
        with self._lock:
            n = self._vectors.shape[0]
            if n == 0:
                return []
            top_k = min(top_k, n)

            if self._faiss_index is not None:
                scores, idx = self._faiss_index.search(query_vector.reshape(1, -1).astype(np.float32), top_k)
                pairs = [(self._records[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1]
                return pairs

            # Brute-force cosine similarity (vectors are pre-normalized, so
            # this is just a dot product).
            sims = self._vectors @ query_vector.astype(np.float32)
            top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return [(self._records[i], float(sims[i])) for i in top_idx]

    # --------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        with self._lock:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            np.save(p / "vectors.npy", self._vectors)
            meta = [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "text": r.text,
                    "index": r.index,
                    "metadata": r.metadata,
                }
                for r in self._records
            ]
            tmp = p / "records.json.tmp"
            tmp.write_text(json.dumps(meta), encoding="utf-8")
            os.replace(tmp, p / "records.json")  # atomic on POSIX

    @classmethod
    def load(cls, path: str, dim: int, backend: str = "numpy") -> "VectorStore":
        store = cls(dim=dim, backend=backend)
        p = Path(path)
        vec_file, meta_file = p / "vectors.npy", p / "records.json"
        if not (vec_file.exists() and meta_file.exists()):
            return store
        vectors = np.load(vec_file)
        records_raw = json.loads(meta_file.read_text(encoding="utf-8"))
        records = [
            VectorRecord(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                index=r["index"],
                metadata=r.get("metadata", {}),
            )
            for r in records_raw
        ]
        if vectors.shape[0]:
            store.add(vectors, records)
        return store

    def __len__(self) -> int:
        return self._vectors.shape[0]
