"""End-to-end FatwaRetriever over a small synthetic index on disk.

score_fatwas, RRF, BM25 and MMR are each unit-tested, but nothing exercised them
wired together through FatwaRetriever.search — which is where index loading,
row/id mapping, hybrid fusion and diversification actually meet. The embedder is
stubbed so no model is downloaded.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app import retrieval as retrieval_module
from app.retrieval import FatwaRetriever

DIM = 8


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


# Four fatwas. 0 and 1 are near-duplicates (the corpus really has 1.5% of these);
# 2 shares no dense direction but carries a distinctive lexical term.
VECTORS = [
    _unit([1, 0, 0, 0, 0, 0, 0, 0]),
    _unit([0.99, 0.14, 0, 0, 0, 0, 0, 0]),
    _unit([0, 1, 0, 0, 0, 0, 0, 0]),
    _unit([0, 0, 1, 0, 0, 0, 0, 0]),
]

ROWS = [
    {"id": 101, "question": "حكم الأغاني", "title": "الأغاني",
     "answer": "الأغاني محرمة.", "link": "https://x/101", "categories": ["أ"]},
    {"id": 102, "question": "حكم المعازف", "title": "المعازف",
     "answer": "المعازف لا تجوز.", "link": "https://x/102", "categories": ["أ"]},
    {"id": 103, "question": "حكم الزكاة", "title": "الزكاة",
     "answer": "الزكاة ركن من أركان الإسلام.", "link": "https://x/103", "categories": ["ب"]},
    {"id": 104, "question": "حكم الحج", "title": "الحج",
     "answer": "الحج فريضة على المستطيع مرة في العمر.", "link": "https://x/104",
     "categories": ["ج"]},
]


@pytest.fixture
def retriever(tmp_path, monkeypatch):
    idx = tmp_path / "index"
    idx.mkdir()

    q_emb = np.stack(VECTORS).astype(np.float16)
    np.save(idx / "question_emb.npy", q_emb)
    np.save(idx / "answer_emb.npy", np.zeros((0, DIM), dtype=np.float16))
    np.save(idx / "answer_parent.npy", np.zeros((0,), dtype=np.int32))
    pd.DataFrame(ROWS).to_parquet(idx / "fatwas_meta.parquet", index=False)
    (idx / "index_manifest.json").write_text(
        json.dumps({"normalize_version": "1.0", "n_fatwas": len(ROWS),
                    "n_chunks": 0, "has_answer_index": False}),
        encoding="utf-8",
    )

    for name, path in [
        ("QUESTION_EMB_PATH", idx / "question_emb.npy"),
        ("ANSWER_EMB_PATH", idx / "answer_emb.npy"),
        ("ANSWER_PARENT_PATH", idx / "answer_parent.npy"),
        ("META_PATH", idx / "fatwas_meta.parquet"),
        ("MANIFEST_PATH", idx / "index_manifest.json"),
    ]:
        monkeypatch.setattr(retrieval_module, name, path)

    r = FatwaRetriever()
    # Stub the embedder: query text maps to a fixed direction, no model needed.
    monkeypatch.setattr(r, "embed", lambda text: np.stack([_unit([1, 0, 0, 0, 0, 0, 0, 0])]))
    return r


def test_search_returns_meta_fields(retriever):
    hits = retriever.search("حكم الأغاني", top_k=2)
    assert len(hits) == 2
    for h in hits:
        assert set(h) >= {"id", "question", "title", "answer", "link",
                          "categories", "similarity"}


def test_categories_survive_as_a_list_of_strings(retriever):
    """Regression: categories used to arrive as a stringified list and the API
    always returned None."""
    hit = retriever.search("حكم الأغاني", top_k=1)[0]
    assert isinstance(hit["categories"], list)
    assert all(isinstance(c, str) for c in hit["categories"])


def test_ids_map_to_the_right_rows(retriever):
    hit = retriever.search("حكم الأغاني", top_k=1)[0]
    assert hit["id"] == 101
    assert hit["link"] == "https://x/101"


def test_top_k_is_respected_and_clamped(retriever):
    assert len(retriever.search("س", top_k=1)) == 1
    assert len(retriever.search("س", top_k=99)) == len(ROWS)


def test_mmr_demotes_the_near_duplicate(retriever, monkeypatch):
    """Fatwas 0 and 1 sit at cosine ~0.99. With MMR on, the second slot should
    not be the twin."""
    monkeypatch.setattr(retrieval_module, "USE_MMR", True)
    monkeypatch.setattr(retrieval_module, "USE_HYBRID", False)
    monkeypatch.setattr(retrieval_module, "MMR_LAMBDA", 0.3)
    ids = [h["id"] for h in retriever.search("حكم الأغاني", top_k=2)]
    assert ids[0] == 101
    assert ids[1] != 102


def test_without_mmr_the_near_duplicate_ranks_second(retriever, monkeypatch):
    """Confirms the previous test is measuring MMR and not something else."""
    monkeypatch.setattr(retrieval_module, "USE_MMR", False)
    monkeypatch.setattr(retrieval_module, "USE_HYBRID", False)
    ids = [h["id"] for h in retriever.search("حكم الأغاني", top_k=2)]
    assert ids == [101, 102]


def test_hybrid_surfaces_a_lexical_only_match(retriever, monkeypatch):
    """The query direction is orthogonal to fatwa 104, so dense alone cannot
    rank it — BM25 must pull it in through RRF."""
    monkeypatch.setattr(retrieval_module, "USE_HYBRID", True)
    monkeypatch.setattr(retrieval_module, "USE_MMR", False)
    monkeypatch.setattr(retrieval_module, "LEXICAL_WEIGHT", 5.0)
    ids = [h["id"] for h in retriever.search("الحج فريضة على المستطيع", top_k=2)]
    assert 104 in ids


def test_dense_only_cannot_surface_that_match(retriever, monkeypatch):
    monkeypatch.setattr(retrieval_module, "USE_HYBRID", False)
    monkeypatch.setattr(retrieval_module, "USE_MMR", False)
    ids = [h["id"] for h in retriever.search("الحج فريضة على المستطيع", top_k=2)]
    assert 104 not in ids


def test_bm25_index_is_built_lazily(retriever):
    assert retriever._bm25 is None
    _ = retriever.bm25
    assert retriever._bm25 is not None


def test_mismatched_index_and_meta_is_rejected(tmp_path, monkeypatch):
    """A stale index against fresh meta must fail loudly, not silently return
    the wrong fatwa for every query."""
    idx = tmp_path / "bad"
    idx.mkdir()
    np.save(idx / "question_emb.npy", np.zeros((2, DIM), dtype=np.float16))
    np.save(idx / "answer_emb.npy", np.zeros((0, DIM), dtype=np.float16))
    np.save(idx / "answer_parent.npy", np.zeros((0,), dtype=np.int32))
    pd.DataFrame(ROWS).to_parquet(idx / "fatwas_meta.parquet", index=False)

    for name, path in [
        ("QUESTION_EMB_PATH", idx / "question_emb.npy"),
        ("ANSWER_EMB_PATH", idx / "answer_emb.npy"),
        ("ANSWER_PARENT_PATH", idx / "answer_parent.npy"),
        ("META_PATH", idx / "fatwas_meta.parquet"),
        ("MANIFEST_PATH", idx / "missing.json"),
    ]:
        monkeypatch.setattr(retrieval_module, name, path)

    with pytest.raises(RuntimeError, match="meta rows"):
        FatwaRetriever()


def test_missing_index_files_raise_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval_module, "QUESTION_EMB_PATH", tmp_path / "nope.npy")
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        FatwaRetriever()
