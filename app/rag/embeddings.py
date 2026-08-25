"""
Embedding providers.

`EmbeddingProvider` is an interface so the vector representation can be
swapped (local model vs. hosted API) without touching retrieval or storage
code. `HashEmbedder` requires no model download and no network call: it's a
deterministic bag-of-character-ngrams hashed into a fixed-size vector. It is
*not* semantically strong, but it makes the whole system runnable and
testable offline, and is a legitimate small-scale/edge deployment choice.
`SentenceTransformerEmbedder` is the recommended provider for real semantic
quality once the dependency and model weights are available.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashEmbedder(EmbeddingProvider):
    """Deterministic, dependency-free, offline-capable embedder.

    Each word is hashed into a bucket with a sign, à la feature hashing /
    the "hashing trick". Cosine similarity over these vectors approximates
    lexical (bag-of-words) overlap — good enough for keyword-heavy corpora
    and, crucially, requires nothing beyond numpy.
    """

    _WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _hash_token(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._WORD_RE.findall(text.lower())
            for tok in tokens:
                bucket, sign = self._hash_token(tok)
                out[i, bucket] += sign
            # Bigrams add a little local word-order sensitivity.
            for a, b in zip(tokens, tokens[1:]):
                bucket, sign = self._hash_token(a + "_" + b)
                out[i, bucket] += 0.5 * sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Wraps `sentence-transformers`. Imported lazily so the dependency is
    only required when this provider is actually selected."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or set "
                "EMBEDDING_PROVIDER=hash to use the offline fallback."
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.astype(np.float32)


def build_embedder(provider: str, model_name: str, dim: int) -> EmbeddingProvider:
    if provider == "hash":
        return HashEmbedder(dim=dim)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unknown embedding provider: {provider}")
