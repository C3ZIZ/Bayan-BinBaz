"""Rank fusion and diversification.

Reciprocal Rank Fusion combines the dense and lexical rankings without needing
their scores to be on a comparable scale — which matters here because BGE-M3
cosine values are heavily compressed (a fatwa's nearest neighbour among 18k
sits at a median cosine of 0.789), while BM25 scores are unbounded.

MMR then removes near-duplicates: 1.5% of this corpus has a twin at >=0.99
similarity, so an undiversified top-5 can waste slots on copies of one fatwa.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[int]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[int]:
    """Fuse ranked id lists into one. Higher-ranked and more widely agreed-upon
    ids come first.

    ``k`` damps the influence of top ranks (the standard RRF constant is 60).
    """
    if weights is not None and len(weights) != len(rank_lists):
        raise ValueError("weights must have one entry per rank list")

    scores: Dict[int, float] = {}
    first_seen: Dict[int, int] = {}
    order = 0
    for list_idx, ids in enumerate(rank_lists):
        w = 1.0 if weights is None else float(weights[list_idx])
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1

    # Sort by score, then by first-seen order so ties are deterministic.
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))


def mmr_select(
    candidate_ids: Sequence[int],
    vectors: np.ndarray,
    relevance: Dict[int, float],
    top_n: int,
    lam: float = 0.7,
) -> List[int]:
    """Maximal Marginal Relevance.

    ``vectors`` is indexed positionally by candidate order (row i belongs to
    ``candidate_ids[i]``) and must be L2-normalized. ``lam`` = 1.0 is pure
    relevance; lower values trade relevance for diversity.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be within [0, 1]")

    candidates = list(candidate_ids)
    if not candidates:
        return []

    vecs = np.asarray(vectors, dtype=np.float32)
    pos = {doc_id: i for i, doc_id in enumerate(candidates)}

    selected: List[int] = []
    remaining = candidates[:]

    while remaining and len(selected) < top_n:
        if not selected:
            best = max(remaining, key=lambda d: relevance.get(d, 0.0))
        else:
            sel_rows = vecs[[pos[d] for d in selected]]

            def score(doc_id: int) -> float:
                redundancy = float(np.max(sel_rows @ vecs[pos[doc_id]]))
                return lam * relevance.get(doc_id, 0.0) - (1.0 - lam) * redundancy

            best = max(remaining, key=score)

        selected.append(best)
        remaining.remove(best)

    return selected
