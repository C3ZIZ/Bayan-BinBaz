"""Compare local GGUF candidates on the two jobs this app actually needs:
structured JSON gate verdicts, and grounded Arabic generation with [n] citations.

Measures what matters, not perplexity: does it answer in Arabic, does it emit
valid JSON, does it cite, and how fast."""
import glob, json, os, re, sys, time
sys.path.insert(0, ".")
from llama_cpp import Llama
from app.gate import GATE_SYSTEM_PROMPT, build_gate_prompt, parse_gate_response
from app.llm import SYSTEM_PROMPT, build_derived_prompt

HITS = [
    {"id": 1, "question": "حكم استماع الأغاني وكتابة أشعارها",
     "answer": "استماع الأغاني المصحوبة بالمعازف لا يجوز، لما فيها من الصد عن ذكر الله وعن الصلاة."},
    {"id": 2, "question": "حكم سماع الأغاني الوطنية المصحوبة بالموسيقى",
     "answer": "الأناشيد التي لا معازف فيها لا بأس بها، أما المصحوبة بالمعازف فلا تجوز."},
]
GATE_CASES = [
    ("ودي اعرف حكم الاغاني وش رايك فيها", "answer"),
    ("اذا جا الحج ورمضان في نفس الوقت نصوم ولا نحج؟", "abstain"),
    ("كيف أطبخ الكبسة السعودية؟", "abstain"),
]

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


def arabic_ratio(t):
    a, l = len(ARABIC.findall(t)), len(LATIN.findall(t))
    return a / max(1, a + l)


def bench(name, path, n_ctx=4096):
    print(f"\n{'='*66}\n{name}  ({os.path.getsize(path)/1e6:.0f} MB)\n{'='*66}", flush=True)
    t = time.time()
    llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=8, verbose=False)
    print(f"  load: {time.time()-t:.1f}s", flush=True)

    # --- gate: structured JSON ---
    ok_json = ok_verdict = 0
    for q, expect in GATE_CASES:
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": GATE_SYSTEM_PROMPT},
                          {"role": "user", "content": build_gate_prompt(q, HITS)}],
                max_tokens=400, temperature=0.0)
            raw = out["choices"][0]["message"]["content"]
            g = parse_gate_response(raw, 2)
            valid = not g.failed_closed
            ok_json += valid
            got = "answer" if g.verdict in ("direct", "derived") else "abstain"
            ok_verdict += got == expect
            print(f"    gate[{expect:7s}] json={'Y' if valid else 'N'} verdict={g.verdict} premise={g.premise_sound}", flush=True)
        except Exception as e:
            print(f"    gate[{expect}] EXC {str(e)[:70]}", flush=True)
    print(f"  GATE: valid_json {ok_json}/{len(GATE_CASES)}  correct {ok_verdict}/{len(GATE_CASES)}")

    # --- generation: Arabic + citations ---
    t = time.time()
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_derived_prompt("ودي اعرف حكم الاغاني وش رايك فيها", HITS)}],
        max_tokens=400, temperature=0.1)
    dt = time.time() - t
    txt = out["choices"][0]["message"]["content"]
    ntok = out["usage"]["completion_tokens"]
    ar = arabic_ratio(txt)
    print(f"  GEN: {ntok} tok in {dt:.1f}s ({ntok/dt:.1f} tok/s)")
    print(f"       arabic_ratio={ar:.2f}  cites={bool(re.search(r'\[\d+\]', txt))}")
    print(f"       --- sample ---\n{txt[:420]}")
    del llm
    return {"gate_json": ok_json, "gate_correct": ok_verdict, "tok_s": ntok/dt, "arabic": ar}


if __name__ == "__main__":
    cands = []
    for pat, label in [
        ("**/ALLaM*Q4_K_M.gguf", "ALLaM-7B-Instruct (SDAIA, Arabic-native)"),
    ]:
        hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{pat}"), recursive=True)
        if hits:
            cands.append((label, hits[0]))
    results = {}
    for label, path in cands:
        try:
            results[label] = bench(label, path)
        except Exception as e:
            print(f"  FAILED: {str(e)[:160]}")
    print(f"\n{'='*66}\nSUMMARY\n{'='*66}")
    for k, v in results.items():
        print(f"  {k[:40]:42s} json={v['gate_json']}/3 verdict={v['gate_correct']}/3 "
              f"{v['tok_s']:5.1f}tok/s arabic={v['arabic']:.2f}")
