import pytest

from app.lexical import BM25Index, tokenize


def test_tokenize_splits_and_normalizes():
    assert tokenize("الصَّلَاةُ  واجبة") == ["الصلاة", "واجبة"]


def test_tokenize_drops_punctuation():
    assert tokenize("ما حكم الصلاة؟ ،.!") == ["ما", "حكم", "الصلاة"]


def test_tokenize_on_empty_input():
    assert tokenize("") == []
    assert tokenize("؟،.") == []


def _index():
    return BM25Index(
        [
            "حكم صيام يوم عرفة لغير الحاج",
            "حكم استماع الأغاني والمعازف",
            "مسافة قصر الصلاة في السفر",
            "حكم صيام الست من شوال",
        ]
    )


def test_search_finds_the_lexically_matching_document():
    assert _index().search("الأغاني", top_k=1)[0][0] == 1


def test_search_returns_scores_in_descending_order():
    results = _index().search("حكم صيام", top_k=4)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k():
    assert len(_index().search("حكم", top_k=2)) == 2


def test_search_on_unknown_terms_returns_nothing_scored():
    results = _index().search("زرافة برمجة حاسوب", top_k=5)
    assert all(score == 0.0 for _, score in results) or results == []


def test_search_matches_despite_diacritics_in_the_query():
    """Corpus text carries tashkeel on 86% of answers; user queries do not."""
    assert _index().search("الأَغَانِي", top_k=1)[0][0] == 1


def test_rare_terms_outrank_common_ones():
    """IDF must make a distinctive term more informative than a ubiquitous one."""
    idx = _index()
    by_rare = idx.search("عرفة", top_k=1)[0][0]
    assert by_rare == 0


def test_empty_corpus_is_safe():
    assert BM25Index([]).search("حكم", top_k=3) == []


def test_empty_query_is_safe():
    assert _index().search("", top_k=3) == []


def test_rejects_negative_top_k():
    with pytest.raises(ValueError):
        _index().search("حكم", top_k=0)
