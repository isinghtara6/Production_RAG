import shutil
import tempfile

import pytest

from app.core.exceptions import ChecksumMismatchError
from app.core.security import sha256_hex
from app.rag.document_store import DocumentStore
from app.rag.embeddings import HashEmbedder
from app.rag.generator import ExtractiveGenerator
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import VectorStore


@pytest.fixture
def pipeline():
    tmpdir = tempfile.mkdtemp()
    embedder = HashEmbedder(dim=128)
    vector_store = VectorStore(dim=128)
    document_store = DocumentStore(f"{tmpdir}/meta.sqlite3")
    generator = ExtractiveGenerator()
    p = RagPipeline(
        embedder=embedder, vector_store=vector_store, document_store=document_store, generator=generator,
        chunk_size_tokens=20, chunk_overlap_tokens=5, default_top_k=3, min_relevance_score=0.0,
        vector_store_path=f"{tmpdir}/vs",
    )
    yield p
    shutil.rmtree(tmpdir)


def test_ingest_then_query_returns_relevant_chunk(pipeline):
    pipeline.ingest(
        document_id="doc1", title="Cats", content="Cats are small domesticated carnivorous mammals.",
        content_sha256_claim=None, metadata={}, source=None,
    )
    pipeline.ingest(
        document_id="doc2", title="Cars", content="Cars are wheeled motor vehicles used for transportation.",
        content_sha256_claim=None, metadata={}, source=None,
    )
    result = pipeline.query(query="Tell me about cats", top_k=2, min_score=None, request_id="req_test")
    assert result["retrieved_count"] > 0
    assert result["sources"][0]["document_id"] == "doc1"


def test_ingest_idempotent_by_content(pipeline):
    r1 = pipeline.ingest(
        document_id="doc1", title="X", content="identical content here",
        content_sha256_claim=None, metadata={}, source=None,
    )
    r2 = pipeline.ingest(
        document_id="doc1-again", title="X", content="identical content here",
        content_sha256_claim=None, metadata={}, source=None,
    )
    assert r1["status"] == "created"
    assert r2["status"] == "already_exists"
    assert r2["document_id"] == r1["document_id"]


def test_ingest_rejects_checksum_mismatch(pipeline):
    with pytest.raises(ChecksumMismatchError):
        pipeline.ingest(
            document_id="doc1", title="X", content="some content",
            content_sha256_claim="0" * 64, metadata={}, source=None,
        )


def test_ingest_accepts_correct_checksum(pipeline):
    content = "verified content"
    checksum = sha256_hex(content.encode("utf-8"))
    result = pipeline.ingest(
        document_id="doc1", title="X", content=content,
        content_sha256_claim=checksum, metadata={}, source=None,
    )
    assert result["content_sha256"] == checksum


def test_delete_document_removes_vectors(pipeline):
    pipeline.ingest(
        document_id="doc1", title="X", content="some content to delete",
        content_sha256_claim=None, metadata={}, source=None,
    )
    pipeline.delete_document("doc1")
    result = pipeline.query(query="some content", top_k=3, min_score=None, request_id="req_test")
    assert result["retrieved_count"] == 0


def test_reingest_same_document_id_updates(pipeline):
    r1 = pipeline.ingest(
        document_id="doc1", title="X", content="original content version one",
        content_sha256_claim=None, metadata={}, source=None,
    )
    r2 = pipeline.ingest(
        document_id="doc1", title="X", content="totally different content version two here",
        content_sha256_claim=None, metadata={}, source=None,
    )
    assert r1["status"] == "created"
    assert r2["status"] == "updated"
