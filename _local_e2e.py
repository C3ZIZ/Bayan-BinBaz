"""End-to-end with LLM_BACKEND=local: real retrieval, real local model, no API.

Verifies the three things the user asked for:
  1. the false-premise question is refused (the reported bug)
  2. a no-exact-match question is ANSWERED, with citations
  3. the verdict is `derived` (not `direct`) when there is no exact match
"""
import os, re, sys

os.environ["LLM_BACKEND"] = "local"
os.environ.pop("HF_TOKEN", None)  # prove the local path needs no Inference quota
sys.path.insert(0, ".")

from app.api import _headline
from app.gate import run_gate
from app.llm import generate_answer
from app.retrieval import get_retriever
from app.citations import strip_invalid_markers

CASES = [
    ("FALSE PREMISE (reported bug)", "اذا جا الحج ورمضان في نفس الوقت نصوم ولا نحج؟"),
    ("NO EXACT MATCH (must answer)", "ودي اعرف حكم الاغاني وش رايك فيها"),
    ("NO EXACT MATCH (dialect)", "ثوبي طويل تحت الكعب وش حكمه"),
    ("OUT OF SCOPE", "كيف أطبخ الكبسة السعودية؟"),
]

r = get_retriever()
for label, q in CASES:
    hits = r.search(q, top_k=8)
    gate = run_gate(q, hits)
    sources = [hits[n - 1] for n in gate.cited_ids] if gate.should_answer else []

    print("\n" + "=" * 74)
    print(f"[{label}]  {q}")
    print("=" * 74)
    print(f"verdict={gate.verdict}  premise_sound={gate.premise_sound}  "
          f"cited={gate.cited_ids}  failed_closed={gate.failed_closed}")
    print(f"HEADLINE: {_headline(gate)}")

    try:
        raw = generate_answer(q, sources, gate.verdict, gate.premise_issue)
        answer, used = strip_invalid_markers(raw, n_sources=len(sources))
        print(f"CITED MARKERS IN ANSWER: {used}")
        for n in used:
            print(f"   [{n}] -> {sources[n-1]['title'][:56]}")
        print("ANSWER:", answer[:420].replace("\n", " "))
    except Exception as e:
        print("GEN FAILED:", str(e)[:150])
    sys.stdout.flush()
