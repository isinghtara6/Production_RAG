import shutil
import tempfile

import numpy as np
import pytest

from app.rag.embeddings import HashEmbedder
from app.rag.vector_store import VectorRecord, VectorStore


def test_hash_embedder_is_deterministic_and_normalized():
    emb = HashEmbedder(dim=64)
    v1 = emb.embed_one("the quick brown fox")
    v2 = emb.embed_one("the quick brown fox")
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5


def test_hash_embedder_similar_text_scores_higher_than_unrelated():
    emb = HashEmbedder(dim=128)
    base = emb.embed_one("the quick brown fox jumps over the lazy dog")
    similar = emb.embed_one("a quick brown fox jumped over a lazy dog")
    unrelated = emb.embed_one("stock market prices rose sharply today")
    assert float(base @ similar) > float(base @ unrelated)


def test_vector_store_add_and_search():
    store = VectorStore(dim=8)
    vectors = np.eye(8, dtype=np.float32)[:3]
    records = [
        VectorRecord(chunk_id=f"c{i}", document_id="doc1", text=f"chunk {i}", index=i)
        for i in range(3)
    ]
    store.add(vectors, records)
    assert len(store) == 3

    results = store.search(vectors[0], top_k=1)
    assert results[0][0].chunk_id == "c0"
    assert results[0][1] == pytest.approx(1.0)


def test_vector_store_delete_by_document():
    store = VectorStore(dim=4)
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    records = [
        VectorRecord(chunk_id="a", document_id="doc1", text="a", index=0),
        VectorRecord(chunk_id="b", document_id="doc2", text="b", index=0),
    ]
    store.add(vectors, records)
    removed = store.delete_by_document("doc1")
    assert removed == 1
    assert len(store) == 1
    assert store._records[0].document_id == "doc2"


def test_vector_store_persistence_roundtrip():
    tmpdir = tempfile.mkdtemp()
    try:
        store = VectorStore(dim=4)
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        records = [
            VectorRecord(chunk_id="a", document_id="doc1", text="hello", index=0, metadata={"title": "T"}),
            VectorRecord(chunk_id="b", document_id="doc1", text="world", index=1, metadata={"title": "T"}),
        ]
        store.add(vectors, records)
        store.save(tmpdir)

        loaded = VectorStore.load(tmpdir, dim=4)
        assert len(loaded) == 2
        results = loaded.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
        assert results[0][0].text == "hello"
        assert results[0][0].metadata["title"] == "T"
    finally:
        shutil.rmtree(tmpdir)


def test_empty_store_search_returns_empty():
    store = VectorStore(dim=4)
    assert store.search(np.zeros(4, dtype=np.float32), top_k=5) == []
