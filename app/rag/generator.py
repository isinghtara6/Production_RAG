"""
Generation providers.

`ExtractiveGenerator` needs no external API and no API key: it composes an
answer directly from the highest-scoring retrieved chunks. It exists so the
service is fully functional (and testable end-to-end) with zero external
dependencies, and as a safe fallback if the configured LLM provider is
unreachable. `AnthropicGenerator` / `OpenAIGenerator` call out to a real LLM
for fluent, synthesized answers, grounded via a strict prompt that
instructs the model to answer only from the supplied context and to say so
when the context is insufficient.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.exceptions import GenerationProviderError
from app.core.logging import get_logger
from app.rag.vector_store import VectorRecord

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Cite passages "
    "inline like [1], [2]. If the passages do not contain enough "
    "information to answer, say so plainly instead of guessing."
)


def _build_context_block(chunks: list[tuple[VectorRecord, float]]) -> str:
    lines = []
    for i, (rec, score) in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {rec.document_id}, relevance: {score:.2f})\n{rec.text}")
    return "\n\n".join(lines)


class GenerationProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, query: str, chunks: list[tuple[VectorRecord, float]]) -> str: ...


class ExtractiveGenerator(GenerationProvider):
    name = "extractive"

    def generate(self, query: str, chunks: list[tuple[VectorRecord, float]]) -> str:
        if not chunks:
            return (
                "I couldn't find any relevant passages in the indexed documents "
                "to answer this question."
            )
        lines = [
            "Based on the retrieved passages, here is what's directly supported:\n"
        ]
        for i, (rec, score) in enumerate(chunks, start=1):
            snippet = rec.text.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
            lines.append(f"[{i}] {snippet}")
        lines.append(
            "\n(This is an extractive summary of the top matches, not a "
            "generative synthesis — configure GENERATION_PROVIDER=anthropic "
            "or openai for a fluent, synthesized answer.)"
        )
        return "\n".join(lines)


class AnthropicGenerator(GenerationProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise GenerationProviderError("ANTHROPIC_API_KEY is not configured.")
        try:
            import anthropic
        except ImportError as e:
            raise GenerationProviderError("anthropic package is not installed.") from e
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, query: str, chunks: list[tuple[VectorRecord, float]]) -> str:
        context = _build_context_block(chunks)
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    }
                ],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except Exception as e:  # noqa: BLE001 - surface as a typed service error
            logger.error("anthropic generation failed", exc_info=True)
            raise GenerationProviderError(f"Anthropic API call failed: {e}") from e


class OpenAIGenerator(GenerationProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise GenerationProviderError("OPENAI_API_KEY is not configured.")
        try:
            import openai
        except ImportError as e:
            raise GenerationProviderError("openai package is not installed.") from e
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, query: str, chunks: list[tuple[VectorRecord, float]]) -> str:
        context = _build_context_block(chunks)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            logger.error("openai generation failed", exc_info=True)
            raise GenerationProviderError(f"OpenAI API call failed: {e}") from e


class GeminiGenerator(GenerationProvider):
    """Uses Google's `google-genai` SDK (the current, GA client — the older
    `google-generativeai` package is deprecated). Requires a Gemini API key
    from https://aistudio.google.com/apikey."""

    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise GenerationProviderError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured.")
        try:
            from google import genai
        except ImportError as e:
            raise GenerationProviderError(
                "google-genai package is not installed. Install it with: pip install google-genai"
            ) from e
        self._client = genai.Client(api_key=api_key)
        # Sensible default if the shared GENERATION_MODEL setting was left
        # pointed at a non-Gemini model name (e.g. the Anthropic default).
        self._model = model if model.startswith("gemini") else "gemini-2.5-flash"

    def generate(self, query: str, chunks: list[tuple[VectorRecord, float]]) -> str:
        context = _build_context_block(chunks)
        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}",
            )
            return resp.text or ""
        except Exception as e:  # noqa: BLE001
            logger.error("gemini generation failed", exc_info=True)
            raise GenerationProviderError(f"Gemini API call failed: {e}") from e


def build_generator(
    provider: str, *, model: str, anthropic_api_key: str, openai_api_key: str, gemini_api_key: str = ""
) -> GenerationProvider:
    if provider == "extractive":
        return ExtractiveGenerator()
    if provider == "anthropic":
        return AnthropicGenerator(api_key=anthropic_api_key, model=model)
    if provider == "openai":
        return OpenAIGenerator(api_key=openai_api_key, model=model)
    if provider == "gemini":
        return GeminiGenerator(api_key=gemini_api_key, model=model)
    raise ValueError(f"Unknown generation provider: {provider}")
