import numpy as np

from app.retrieval import score_fatwas


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_alpha_one_uses_only_question_similarity():
    q_emb = np.stack([_unit([1, 0]), _unit([0, 1])]).astype(np.float16)
    a_emb = np.stack([_unit([0, 1]), _unit([1, 0])]).astype(np.float16)
    parent = np.array([0, 1], dtype=np.int32)
    scores = score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=1.0)
    assert scores[0] > scores[1]


def test_alpha_zero_uses_only_answer_similarity():
    q_emb = np.stack([_unit([1, 0]), _unit([0, 1])]).astype(np.float16)
    a_emb = np.stack([_unit([0, 1]), _unit([1, 0])]).astype(np.float16)
    parent = np.array([0, 1], dtype=np.int32)
    scores = score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=0.0)
    assert scores[1] > scores[0]


def test_max_pooling_over_multiple_chunks():
    """A fatwa is scored by its BEST matching chunk, not its average."""
    q_emb = np.stack([_unit([0, 1])]).astype(np.float16)
    a_emb = np.stack([_unit([0, 1]), _unit([1, 0])]).astype(np.float16)
    parent = np.array([0, 0], dtype=np.int32)
    scores = score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=0.0)
    assert np.isclose(scores[0], 1.0, atol=1e-2)


def test_returns_one_score_per_fatwa():
    q_emb = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])]).astype(np.float16)
    a_emb = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])]).astype(np.float16)
    parent = np.array([0, 1, 2], dtype=np.int32)
    assert score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=0.5).shape == (3,)


def test_fatwa_with_no_chunks_is_still_finite():
    q_emb = np.stack([_unit([1, 0]), _unit([0, 1])]).astype(np.float16)
    a_emb = np.stack([_unit([1, 0])]).astype(np.float16)
    parent = np.array([0], dtype=np.int32)
    scores = score_fatwas(_unit([0, 1]), q_emb, a_emb, parent, alpha=0.5)
    assert np.isfinite(scores).all()


def test_empty_answer_index_falls_back_to_questions():
    q_emb = np.stack([_unit([1, 0]), _unit([0, 1])]).astype(np.float16)
    a_emb = np.zeros((0, 2), dtype=np.float16)
    parent = np.zeros((0,), dtype=np.int32)
    scores = score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=0.5)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]


def test_negative_chunk_similarity_does_not_stay_at_zero():
    """max.at seeded with zeros would wrongly floor a genuinely negative
    similarity at 0, inflating unrelated fatwas."""
    q_emb = np.stack([_unit([1, 0])]).astype(np.float16)
    a_emb = np.stack([_unit([-1, 0])]).astype(np.float16)
    parent = np.array([0], dtype=np.int32)
    scores = score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=0.0)
    assert scores[0] < 0
