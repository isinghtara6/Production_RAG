"""
Chunking.

Splits document text into overlapping windows sized in tokens (falling back
to a whitespace tokenizer if `tiktoken` isn't installed, so this module has
zero hard dependencies). Overlap preserves context across chunk boundaries
so an answer-relevant sentence split across two chunks is still retrievable
from at least one of them.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _encode(text: str) -> list[int]:
        return _ENC.encode(text)

    def _decode(tokens: list[int]) -> str:
        return _ENC.decode(tokens)

except Exception:  # pragma: no cover - exercised when tiktoken is absent
    # Whitespace-token fallback: "tokens" are just words. Coarser than a real
    # BPE tokenizer but keeps chunk sizing deterministic and dependency-free.
    def _encode(text: str) -> list[str]:  # type: ignore[misc]
        return text.split()

    def _decode(tokens: list[str]) -> str:  # type: ignore[misc]
        return " ".join(tokens)


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int
    start_token: int
    end_token: int


def chunk_text(text: str, *, chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_size_tokens)")

    text = text.strip()
    if not text:
        return []

    tokens = _encode(text)
    stride = chunk_size_tokens - overlap_tokens

    chunks: list[Chunk] = []
    start = 0
    index = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size_tokens, n)
        window = tokens[start:end]
        chunk_str = _decode(window).strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    index=index,
                    text=chunk_str,
                    token_count=len(window),
                    start_token=start,
                    end_token=end,
                )
            )
            index += 1
        if end == n:
            break
        start += stride

    return chunks
