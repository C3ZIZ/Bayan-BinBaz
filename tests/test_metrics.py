import math

from eval.metrics import aggregate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_hit_at_rank_1():
    assert recall_at_k([5, 2, 9], 5, 1) == 1.0


def test_recall_miss_outside_k():
    assert recall_at_k([5, 2, 9], 9, 2) == 0.0


def test_recall_hit_inside_k():
    assert recall_at_k([5, 2, 9], 9, 3) == 1.0


def test_reciprocal_rank_positions():
    assert reciprocal_rank([5, 2, 9], 5) == 1.0
    assert reciprocal_rank([5, 2, 9], 2) == 0.5
    assert reciprocal_rank([5, 2, 9], 7) == 0.0


def test_ndcg_rank_1_is_one():
    assert ndcg_at_k([5, 2, 9], 5, 3) == 1.0


def test_ndcg_rank_2_matches_formula():
    assert math.isclose(ndcg_at_k([5, 2, 9], 2, 3), 1 / math.log2(3))


def test_ndcg_miss_is_zero():
    assert ndcg_at_k([5, 2, 9], 7, 3) == 0.0


def test_aggregate_reports_all_metrics():
    rows = [
        {"ranked_ids": [1, 2, 3], "relevant_id": 1},
        {"ranked_ids": [4, 5, 6], "relevant_id": 5},
    ]
    out = aggregate(rows, ks=(1, 3))
    assert out["recall@1"] == 0.5
    assert out["recall@3"] == 1.0
    assert math.isclose(out["mrr"], 0.75)
    assert out["n"] == 2


def test_aggregate_on_empty_input():
    out = aggregate([], ks=(1,))
    assert out["n"] == 0
    assert out["recall@1"] == 0.0
    assert out["mrr"] == 0.0
