import pytest

from app.chunking import chunk_text


def wc(s: str) -> int:
    """One 'token' per whitespace word — deterministic and model-free."""
    return len(s.split())


def test_short_text_is_a_single_chunk():
    assert chunk_text("a b c", wc, max_tokens=10, overlap_tokens=2) == ["a b c"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("", wc, max_tokens=10, overlap_tokens=2) == []
    assert chunk_text("   ", wc, max_tokens=10, overlap_tokens=2) == []


def test_long_text_is_split():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, wc, max_tokens=30, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(wc(c) <= 30 for c in chunks)


def test_chunks_overlap():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, wc, max_tokens=30, overlap_tokens=5)
    first_tail = chunks[0].split()[-5:]
    assert any(t in chunks[1].split() for t in first_tail)


def test_no_content_is_lost():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, wc, max_tokens=30, overlap_tokens=5)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == {str(i) for i in range(100)}


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValueError):
        chunk_text("a b c", wc, max_tokens=10, overlap_tokens=10)


def test_terminates_on_a_single_oversized_word():
    # Every single word already exceeds max_tokens; must still terminate and
    # cover the whole input rather than looping forever.
    chunks = chunk_text(
        "aaaa bb cc", lambda s: 50 * len(s.split()), max_tokens=10, overlap_tokens=2
    )
    assert chunks == ["aaaa", "bb", "cc"]


def test_arabic_text_splits_on_word_boundaries():
    text = " ".join(["الصلاة"] * 60)
    chunks = chunk_text(text, wc, max_tokens=20, overlap_tokens=4)
    assert len(chunks) > 1
    for c in chunks:
        assert "الصلاة" in c
        assert c == c.strip()


def test_defaults_are_internally_consistent():
    # The production defaults must not trip the overlap validation.
    assert chunk_text("كلمة", wc) == ["كلمة"]
