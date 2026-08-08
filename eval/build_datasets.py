"""Generate the golden retrieval set from the corpus.

Two mechanical variants per sampled fatwa keep the set reproducible and free of
hand-labelling:

  verbatim   - the fatwa's own question, normalized. Tests that the index can
               find a document from its exact text.
  paraphrase - the question stripped of the transcription openers that pad the
               corpus ("يقول السائل:", "أحسن الله إليكم") and truncated, which
               approximates how a user actually types. Tests robustness to the
               phrasing gap that real traffic has.

Disjoint fatwas are used for the two slices so a paraphrase never competes with
its own verbatim twin.
"""

import sys
from pathlib import Path as _Path

# Entry-point scripts are run directly (python eval/x.py), so the repo
# root is not on sys.path by default.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import re
from pathlib import Path

import pandas as pd

from app.normalize import normalize_arabic

META = Path("data/index/fatwas_meta.parquet")
OUT = Path("eval/datasets/golden.jsonl")
SEED = 20260808

# Written against the POST-normalization forms. normalize_arabic folds أ/إ/آ/ٱ
# to ا but leaves ؤ and ئ alone, so "يسأل" becomes "يسال" while "السؤال" and
# "السائل" keep their hamza — both spellings must be listed.
_LEAD = re.compile(
    r"^(?:يقول|تقول|يسال|تسال|يسأل|السؤال|السوال|سؤال|سوال|"
    r"احسن الله اليكم|بارك الله فيكم|جزاكم الله خيرا|وفقكم الله|"
    r"فضيلة الشيخ|سماحة الشيخ|"
    r"هذا السائل يقول|هذه السائلة تقول|السائل يقول|السائلة تقول|"
    r"السائل|السائلة|المستمع|المستمعة|"
    r"المقدم|الشيخ)\s*[:،.]?\s*"
)


def paraphrase(question: str) -> str:
    """Strip stacked transcription openers, then keep the first 18 words."""
    q = normalize_arabic(question)
    prev = None
    while prev != q:
        prev = q
        q = _LEAD.sub("", q).strip()
    words = q.split()
    if not words:
        return normalize_arabic(question)
    return " ".join(words[:18])


def main():
    meta = pd.read_parquet(META)
    long_enough = meta[meta["question"].astype(str).str.split().str.len() >= 8]
    if len(long_enough) < 200:
        raise RuntimeError(f"only {len(long_enough)} usable questions; need 200")

    sample = long_enough.sample(200, random_state=SEED)
    verbatim_src = sample.iloc[:100]
    para_src = sample.iloc[100:]

    rows, seen = [], set()

    for _, r in verbatim_src.iterrows():
        q = normalize_arabic(str(r["question"]))
        if not q or q in seen:
            continue
        seen.add(q)
        rows.append({"query": q, "relevant_id": int(r["id"]), "kind": "verbatim"})

    for _, r in para_src.iterrows():
        q = paraphrase(str(r["question"]))
        if not q or q in seen:
            continue
        seen.add(q)
        rows.append({"query": q, "relevant_id": int(r["id"]), "kind": "paraphrase"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in ("verbatim", "paraphrase")}
    print(f"wrote {len(rows)} rows to {OUT}  {kinds}")


if __name__ == "__main__":
    main()
