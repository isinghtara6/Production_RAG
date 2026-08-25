from app.rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size_tokens=10, overlap_tokens=2) == []


def test_short_text_single_chunk():
    chunks = chunk_text("hello world", chunk_size_tokens=10, overlap_tokens=2)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0


def test_long_text_produces_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    # consecutive chunks must overlap by exactly `overlap_tokens`
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_token == a.end_token - 5
    # union of chunk token ranges must cover the whole document
    assert chunks[-1].end_token == 100
    assert chunks[0].start_token == 0


def test_indices_are_sequential():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size_tokens=10, overlap_tokens=3)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_invalid_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size_tokens=10, overlap_tokens=-1)
