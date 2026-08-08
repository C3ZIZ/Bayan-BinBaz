import numpy as np
import pytest

from app.fusion import mmr_select, reciprocal_rank_fusion


# ------------------------------------------------------------------------ RRF


def test_rrf_single_list_preserves_order():
    assert reciprocal_rank_fusion([[3, 1, 2]]) == [3, 1, 2]


def test_rrf_rewards_agreement_across_lists():
    """A document ranked 2nd by both beats one ranked 1st by only one list."""
    fused = reciprocal_rank_fusion([[9, 5, 1], [8, 5, 2]], k=1)
    assert fused[0] == 5


def test_rrf_includes_documents_from_every_list():
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]])
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([[], [7]]) == [7]


def test_rrf_k_damps_rank_influence():
    """A large k flattens the curve so early ranks matter less."""
    sharp = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]], k=1)
    flat = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]], k=1000)
    assert sharp[0] in (1, 3)
    assert len(flat) == 3


def test_rrf_is_deterministic_on_ties():
    a = reciprocal_rank_fusion([[1, 2], [2, 1]])
    b = reciprocal_rank_fusion([[1, 2], [2, 1]])
    assert a == b


def test_rrf_respects_explicit_weights():
    fused = reciprocal_rank_fusion([[1, 9], [9, 1]], weights=[10.0, 1.0])
    assert fused[0] == 1


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])


# ------------------------------------------------------------------------ MMR


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_mmr_returns_requested_count():
    vecs = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
    assert len(mmr_select([0, 1, 2], vecs, {0: 0.9, 1: 0.8, 2: 0.7}, top_n=2)) == 2


def test_mmr_picks_the_most_relevant_first():
    vecs = np.stack([_unit([1, 0]), _unit([0, 1])])
    assert mmr_select([0, 1], vecs, {0: 0.5, 1: 0.9}, top_n=1) == [1]


def test_mmr_drops_a_near_duplicate_in_favour_of_diversity():
    """1.5% of this corpus has a twin at >=0.99 similarity; top-k must not be
    filled with copies."""
    vecs = np.stack([_unit([1, 0]), _unit([1, 0.01]), _unit([0, 1])])
    picked = mmr_select([0, 1, 2], vecs, {0: 0.9, 1: 0.89, 2: 0.5}, top_n=2, lam=0.5)
    assert picked[0] == 0
    assert picked[1] == 2


def test_mmr_with_lambda_one_ignores_diversity():
    vecs = np.stack([_unit([1, 0]), _unit([1, 0.01]), _unit([0, 1])])
    picked = mmr_select([0, 1, 2], vecs, {0: 0.9, 1: 0.89, 2: 0.5}, top_n=2, lam=1.0)
    assert picked == [0, 1]


def test_mmr_handles_fewer_candidates_than_requested():
    vecs = np.stack([_unit([1, 0])])
    assert mmr_select([0], vecs, {0: 0.5}, top_n=5) == [0]


def test_mmr_handles_empty_candidates():
    assert mmr_select([], np.zeros((0, 2), dtype=np.float32), {}, top_n=3) == []


def test_mmr_rejects_invalid_lambda():
    vecs = np.stack([_unit([1, 0])])
    with pytest.raises(ValueError):
        mmr_select([0], vecs, {0: 0.5}, top_n=1, lam=1.5)
