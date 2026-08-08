"""Measure the gate on the two datasets that matter for correctness.

Two failure modes, deliberately measured separately because they trade off
against each other:

  SAFETY  — answering a question the fatwas do not support (adversarial set).
            This is what produced the reported Hajj/Ramadan bug.
  PRODUCT — refusing a question that IS answerable by derivation (derived set).
            A gate tuned only against the adversarial set collapses into
            over-refusal, which would be a worse app than the original.

Both numbers must be reported together. A high abstention rate is only good news
if the false-abstention rate stayed low.
"""

import sys
from pathlib import Path as _Path

# Entry-point scripts are run directly (python eval/x.py), so the repo
# root is not on sys.path by default.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from app.gate import run_gate
from app.retrieval import get_retriever

DERIVED = Path("eval/datasets/derived.jsonl")
ADVERSARIAL = Path("eval/datasets/adversarial.jsonl")
RESULTS_DIR = Path("eval/results")
GATE_CANDIDATES = 8
RETRIES = int(os.getenv("EVAL_GATE_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("EVAL_GATE_BACKOFF", "4"))
PACING = float(os.getenv("EVAL_GATE_PACING", "1.5"))


def load(path: Path, limit: int = 0) -> List[Dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def evaluate(rows: List[Dict], workers: int) -> List[Dict]:
    retriever = get_retriever()

    # Retrieval is serial: BGE-M3 is one shared model and the box is CPU-only.
    # Only the gate's network calls are parallelised.
    prepared = []
    for i, row in enumerate(rows, 1):
        hits = retriever.search(row["query"], top_k=GATE_CANDIDATES)
        prepared.append((row, hits))
        if i % 20 == 0:
            print(f"  retrieved {i}/{len(rows)}", flush=True)

    def judge(item):
        row, hits = item
        # The gate itself fails closed immediately — correct in production, but
        # useless for measurement: a throttled run scores 100% abstention while
        # having judged nothing. The EVAL therefore retries with backoff so the
        # numbers reflect the gate's decisions, not the provider's quota.
        gate = None
        for attempt in range(RETRIES):
            gate = run_gate(row["query"], hits)
            if not gate.failed_closed:
                break
            time.sleep(BACKOFF_BASE * (2**attempt))
        retrieved_ids = [h["id"] for h in hits]
        supporting = set(row.get("supporting_ids") or [])
        return {
            "query": row["query"],
            "expected": row["expected_verdict"],
            "kind": row.get("kind") or row.get("reason", ""),
            "verdict": gate.verdict,
            "premise_sound": gate.premise_sound,
            "premise_issue": gate.premise_issue,
            "failed_closed": gate.failed_closed,
            "cited_ids": gate.cited_ids,
            "reason": gate.reason,
            "support_retrieved": bool(supporting & set(retrieved_ids)) if supporting else None,
        }

    if workers <= 1:
        out = []
        for i, item in enumerate(prepared, 1):
            out.append(judge(item))
            if i % 10 == 0:
                print(f"  judged {i}/{len(prepared)}", flush=True)
            time.sleep(PACING)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(judge, prepared))


def report(derived: List[Dict], adversarial: List[Dict]) -> Dict:
    answered = [r for r in derived if r["verdict"] in ("direct", "derived")]
    false_abstentions = [r for r in derived if r["verdict"] == "abstain"]
    correct_abstentions = [r for r in adversarial if r["verdict"] == "abstain"]
    wrong_answers = [r for r in adversarial if r["verdict"] != "abstain"]

    summary = {
        "derived_n": len(derived),
        "derived_answered": len(answered),
        "false_abstention_rate": len(false_abstentions) / len(derived) if derived else 0.0,
        "adversarial_n": len(adversarial),
        "abstention_recall": len(correct_abstentions) / len(adversarial) if adversarial else 0.0,
        "wrongly_answered": len(wrong_answers),
        "gate_failures": sum(r["failed_closed"] for r in derived + adversarial),
    }

    print("\n" + "=" * 72)
    print("PRODUCT — derived set: questions with NO exact match that MUST be answered")
    print("=" * 72)
    print(f"  answered correctly : {len(answered)}/{len(derived)}")
    print(f"  FALSE ABSTENTIONS  : {len(false_abstentions)}  "
          f"({summary['false_abstention_rate']:.1%})   <- lower is better")
    by_kind = Counter(r["kind"] for r in false_abstentions)
    if by_kind:
        print(f"  by kind            : {dict(by_kind)}")
    unreachable = [r for r in derived if r["support_retrieved"] is False]
    if unreachable:
        print(f"  supporting fatwa NOT in top-{GATE_CANDIDATES}: {len(unreachable)} "
              f"(a retrieval problem, not a gate problem)")

    print("\n" + "=" * 72)
    print("SAFETY — adversarial set: questions that MUST be refused")
    print("=" * 72)
    print(f"  abstained correctly: {len(correct_abstentions)}/{len(adversarial)}  "
          f"({summary['abstention_recall']:.1%})   <- higher is better")
    print(f"  WRONGLY ANSWERED   : {len(wrong_answers)}")
    for r in wrong_answers[:10]:
        print(f"    [{r['kind']}] {r['query'][:64]} -> {r['verdict']}")

    hajj = [r for r in adversarial if "الحج" in r["query"] and "رمضان" in r["query"]]
    if hajj:
        r = hajj[0]
        status = "PASS" if r["verdict"] == "abstain" else "FAIL"
        print(f"\n  reported bug (Hajj+Ramadan): {status} -> verdict={r['verdict']}, "
              f"premise_sound={r['premise_sound']}")
        if r["premise_issue"]:
            print(f"    premise_issue: {r['premise_issue'][:120]}")

    print(f"\n  gate calls that failed closed: {summary['gate_failures']}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="rows per dataset (0 = all)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tag", default="latest")
    a = p.parse_args()

    if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
        raise SystemExit("HF_TOKEN is required to run the gate eval.")

    print("Evaluating derived set...", flush=True)
    derived_rows = evaluate(load(DERIVED, a.limit), a.workers)
    print("Evaluating adversarial set...", flush=True)
    adversarial_rows = evaluate(load(ADVERSARIAL, a.limit), a.workers)

    summary = report(derived_rows, adversarial_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"gate_{a.tag}.json"
    out.write_text(
        json.dumps({"summary": summary, "derived": derived_rows,
                    "adversarial": adversarial_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved -> {out}")
