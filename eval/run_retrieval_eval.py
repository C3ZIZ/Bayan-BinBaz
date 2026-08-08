"""Score retrieval against an eval set and persist the result.

This is the gate for every change to the retrieval stack: run it before and
after, and do not ship a regression.
"""

import argparse
import json
from pathlib import Path

from app.retrieval import get_retriever
from eval.metrics import aggregate

DEFAULT_SET = Path("eval/datasets/golden.jsonl")
RESULTS_DIR = Path("eval/results")


def load_rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(dataset: Path, top_k: int, tag: str):
    rows = load_rows(dataset)
    retriever = get_retriever()

    scored = []
    for i, row in enumerate(rows, 1):
        hits = retriever.search(row["query"], top_k=top_k)
        scored.append(
            {
                "ranked_ids": [h["id"] for h in hits],
                "relevant_id": row["relevant_id"],
                "kind": row.get("kind", "unknown"),
            }
        )
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    overall = aggregate(scored)
    by_kind = {
        kind: aggregate([s for s in scored if s["kind"] == kind])
        for kind in sorted({s["kind"] for s in scored})
    }

    def show(title, m):
        print(f"\n=== {title} ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    show("OVERALL", overall)
    for kind, m in by_kind.items():
        show(kind, m)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{tag}.json"
    out.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "top_k": top_k,
                "alpha": __import__("app.retrieval", fromlist=["ALPHA"]).ALPHA,
                "overall": overall,
                "by_kind": by_kind,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved -> {out}")
    return overall


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_SET)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--tag", default="latest")
    a = p.parse_args()
    run(a.dataset, a.top_k, a.tag)
