import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

INDEX = Path("data/index")

pytestmark = pytest.mark.skipif(
    not (INDEX / "index_manifest.json").exists(),
    reason="index not built yet (run build_index.py)",
)


@pytest.fixture(scope="module")
def artifacts():
    return (
        np.load(INDEX / "question_emb.npy"),
        np.load(INDEX / "answer_emb.npy"),
        np.load(INDEX / "answer_parent.npy"),
        pd.read_parquet(INDEX / "fatwas_meta.parquet"),
        json.loads((INDEX / "index_manifest.json").read_text(encoding="utf-8")),
    )


def test_dtypes_are_float16(artifacts):
    q, a, *_ = artifacts
    assert q.dtype == np.float16
    assert a.dtype == np.float16


def test_question_rows_match_meta_rows(artifacts):
    q, _, _, meta, _ = artifacts
    assert len(q) == len(meta)


def test_parent_map_matches_answer_rows(artifacts):
    _, a, parent, _, _ = artifacts
    assert len(parent) == len(a)


def test_parent_indices_are_in_range(artifacts):
    _, _, parent, meta, _ = artifacts
    assert parent.min() >= 0
    assert parent.max() < len(meta)


def test_every_fatwa_has_at_least_one_chunk(artifacts):
    """No fatwa may be unreachable through the answer field."""
    _, _, parent, meta, _ = artifacts
    assert len(np.unique(parent)) == len(meta)


def test_vectors_are_unit_norm(artifacts):
    q, a, *_ = artifacts
    for m in (q[:200], a[:200]):
        norms = np.linalg.norm(m.astype(np.float32), axis=1)
        assert np.allclose(norms, 1.0, atol=2e-3)


def test_no_literal_nan_answers(artifacts):
    """Regression guard: 72 rows previously had the string "nan" as the answer."""
    *_, meta, _ = artifacts
    bad = (meta["answer"].astype(str).str.strip().str.lower() == "nan").sum()
    assert bad == 0


def test_categories_are_parsed_sequences(artifacts):
    *_, meta, _ = artifacts
    first = meta["categories"].iloc[0]
    assert isinstance(first, (list, np.ndarray))
    assert len(first) > 0


def test_manifest_records_normalize_version(artifacts):
    from app.normalize import NORMALIZE_VERSION

    *_, manifest = artifacts
    assert manifest["normalize_version"] == NORMALIZE_VERSION


def test_manifest_counts_match_arrays(artifacts):
    q, a, _, _, manifest = artifacts
    assert manifest["n_fatwas"] == len(q)
    assert manifest["n_chunks"] == len(a)


def test_chunking_actually_expanded_the_corpus(artifacts):
    """17.7% of fatwas exceeded the old 512-token limit, so chunk count must
    exceed fatwa count — otherwise chunking silently did nothing."""
    q, a, *_ = artifacts
    assert len(a) > len(q)


def test_meta_ids_are_unique(artifacts):
    *_, meta, _ = artifacts
    assert meta["id"].is_unique
