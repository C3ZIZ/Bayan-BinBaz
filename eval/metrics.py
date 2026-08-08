"""Retrieval metrics.

Single-relevant-document formulation: each eval row has exactly one correct
parent fatwa id. Once chunking lands, retrieval returns chunks, so callers must
map chunks to their parent fatwa and deduplicate before ranking — these
functions operate on the resulting fatwa-level rank list.
"""

import math
from typing import Dict, List, Sequence, Tuple


def recall_at_k(ranked_ids: Sequence[int], relevant_id: int, k: int) -> float:
    return 1.0 if relevant_id in list(ranked_ids)[:k] else 0.0


def reciprocal_rank(ranked_ids: Sequence[int], relevant_id: int) -> float:
    ids = list(ranked_ids)
    if relevant_id not in ids:
        return 0.0
    return 1.0 / (ids.index(relevant_id) + 1)


def ndcg_at_k(ranked_ids: Sequence[int], relevant_id: int, k: int) -> float:
    ids = list(ranked_ids)[:k]
    if relevant_id not in ids:
        return 0.0
    # Binary relevance with a single relevant document, so IDCG is 1.
    return 1.0 / math.log2(ids.index(relevant_id) + 2)


def aggregate(
    rows: List[Dict], ks: Tuple[int, ...] = (1, 3, 5, 10, 20)
) -> Dict[str, float]:
    """Aggregate per-query results. ``rows`` entries need ``ranked_ids`` and
    ``relevant_id``."""
    out: Dict[str, float] = {"n": len(rows)}
    if not rows:
        for k in ks:
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        out["mrr"] = 0.0
        return out

    n = len(rows)
    for k in ks:
        out[f"recall@{k}"] = (
            sum(recall_at_k(r["ranked_ids"], r["relevant_id"], k) for r in rows) / n
        )
        out[f"ndcg@{k}"] = (
            sum(ndcg_at_k(r["ranked_ids"], r["relevant_id"], k) for r in rows) / n
        )
    out["mrr"] = sum(reciprocal_rank(r["ranked_ids"], r["relevant_id"]) for r in rows) / n
    return out
