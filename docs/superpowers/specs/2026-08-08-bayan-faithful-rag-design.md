# Bayan — Faithfulness-First RAG: Design

**Date:** 2026-08-08
**Status:** Approved, pending implementation plan

---

## 1. Problem

The assistant fabricates rulings. The reported symptom:

> «اذا جا الحج ورمضان في نفس الوقت نصوم ولا نحج؟»

Ramadan is month 9 and Dhul-Hijjah is month 12, so the scenario cannot occur. The
system answers anyway. Measurement shows three independent causes, none of which
is "the model hallucinated".

### 1.1 Measured evidence

All figures below were measured against the live index (18,756 fatwas, BGE-M3).

**The app cannot say "I don't know."** `FatwaRetriever.search()` always returns
`min(top_k, N)` hits, so `hits_raw` is never empty, so `_retrieve()` never returns
`"none"`. `_header_for("none")` and the `if not hits` prompt branch are unreachable.

**`EXACT_THRESHOLD = 0.90` fires on 18% of *perfect* matches.** Embedding 300 real
fatwa questions and measuring similarity to their own fatwa — the theoretical
ceiling:

| p5 | p25 | p50 | p75 | p95 |
|----|-----|-----|-----|-----|
| 0.679 | 0.753 | **0.815** | 0.885 | 0.943 |

Only 18% clear 0.90. The system is stuck in `approx` mode even when it holds the
correct fatwa. **Consequence for design: the no-exact-match path is the normal
path, not the fallback.**

**No cosine threshold can separate answerable from unanswerable.** Probing the
live index:

| query | top-1 sim |
|-------|-----------|
| «إذا جا الحج ورمضان في نفس الوقت…» (impossible) | **0.628** |
| «ما حكم صيام يوم عرفة لغير الحاج؟» (valid) | 0.673 |
| «ودي اعرف حكم الاغاني وش رايك» (valid, dialect) | **0.499** |
| «كيف أطبخ الكبسة السعودية؟» (out of scope) | 0.473 |
| gibberish | 0.450 |

In-scope minimum (0.499) sits **below** bad-query maximum (0.628). The
distributions overlap. A threshold-based gate is mathematically incapable of
fixing this.

**The prompt forbids refusal.** `build_approx_prompt` instructs
«استخرج من الفتاوى السابقة ما يساعد على توجيه السائل» with no escape hatch and no
instruction to test the question's premise.

**19.6% of the corpus is not indexed.** `build_index.py` sets `max_length=512`;
BGE-M3 supports 8192. Measured with the real tokenizer:

| max_length | docs truncated | corpus tokens indexed |
|------------|----------------|------------------------|
| **512 (current)** | **3,315 (17.7%)** | **80.4% — 19.6% lost** |
| 1024 | 736 (3.9%) | 93.9% |
| 2048 | 98 (0.5%) | 98.7% |

**Query/document asymmetry.** The index embeds `question + "\n" + answer`, of
which **81% of tokens are the answer**. Users send short questions. This is what
compresses self-similarity to 0.815.

**Supporting data defects.** 72 rows have the literal string `"nan"` as their
answer (`.astype(str)` runs before `dropna`). `categories` is stored as the string
`"['التفسير']"` and never parsed, so `_categories()` always returns `None` —
215 categories at 100% coverage, unused. 86% of answers carry diacritics, user
queries carry ~0%; stripping tashkeel cuts tokens 10.3%. 1.5% of fatwas have a
near-twin at ≥0.99 similarity, with no dedup before top-k.

### 1.2 What already works — do not regress it

- Retrieval recall is excellent: **recall@1 = 97.7%, recall@5 = 100%, MRR = 0.988**.
- Embeddings are correctly L2-normalized (verified: all norms = 1.0000).
- Brute-force NumPy scan is the right choice at this scale; do not introduce FAISS.
- Lazy model load + idle unload + `malloc_trim` is a sound ~4GB VPS design.
- SSE streaming works end to end.
- Frontend is XSS-safe (`textContent` throughout).

---

## 2. Goals and non-goals

### Goals

1. **No unsupported assertion.** Every ruling in an answer traces to a cited fatwa.
2. **Answer correctly when there is no exact match.** Absence of a verbatim match
   must not produce a refusal. This is the primary quality path.
3. **Never misattribute.** A ruling derived by application must be visibly
   distinguished from a ruling Ibn Baz stated directly.
4. **Recover the unindexed 19.6%** of the corpus.
5. **Make behavior measurable** — no change ships without a metric.

### Non-goals

- Multi-turn conversation memory. Deferred; tracked as a known defect.
- Replacing NumPy retrieval with a vector database.
- Changing the hosted-LLM deployment model or the ~4GB RAM budget.
- Re-scraping or expanding the corpus.

---

## 3. Design principle

> The system's default is **not to assert**. An answer is a claim about what
> Shaykh Ibn Baz ruled, and every such claim must be attributable to a specific
> retrieved fatwa.

The Hajj/Ramadan bug is a *symptom* of unsupported assertion, not a separate
false-premise feature. Fixing the general rule fixes the specific case.

---

## 4. Architecture

```
question
  ↓
[1] normalize        tashkeel, alef/ta-marbuta, tatweel
  ↓
[2] hybrid retrieval dense + sparse → RRF → top-50
  ↓
[3] MMR dedup        → top-8 candidates
  ↓
[4] GATE             hosted LLM, structured JSON, ~1s, NOT streamed
                     in:  question + 8 (question, snippet) pairs
                     out: {premise_sound, premise_issue, verdict, cited_ids[], reasoning}
  ↓
[5] generation       streamed, context = ONLY gate-approved fatwas
  ↓
[6] UI               [n] markers → clickable fatwa links + verdict badge
```

**Where faithfulness is enforced:** step 5's context contains only the fatwas the
gate approved. The model cannot cite what it was never given, and is instructed
that every ruling must carry a marker.

**Why the gate is a hosted LLM call:** the measured overlap in §1.1 rules out a
threshold. A local cross-encoder would work but costs ~600MB–2.3GB alongside
BGE-M3 on a 4GB box. The hosted gate costs zero local RAM and one small API call.

---

## 5. The three verdicts

`derived` is the expected common case (see §1.1: exact matches are rare). It must
be the best-built path, not a degraded one.

### `direct`
One or more retrieved fatwas address the question itself.
Answer from them. Cite each ruling `[n]`.

### `derived` — the primary path
No fatwa addresses this exact case, but the retrieved fatwas establish the ruling
or the governing principle.

Answer, cite, and **structurally separate two things**:

1. what the shaykh stated — «قرر الشيخ ابن باز أن…» `[1]`
2. its application to the asker — «وعليه، فإن حالتك…»

The answer opens by stating no fatwa addresses this case exactly, and closes
recommending a scholar. Faithfulness rules here are *stricter* than `direct`:
the application sentence may not introduce any ruling not present in a cited
fatwa.

### `abstain`
Retrieved fatwas do not establish an answer, **or** the question rests on a false
premise, **or** the question is out of scope.

Cite nothing. Assert no ruling. When `premise_sound == false`, state the defect
plainly and stop — this is the Hajj/Ramadan path.

### Failure asymmetry

Wrongly abstaining on an answerable question is a **product failure**.
Wrongly answering an unanswerable one is a **safety failure**.
Both are measured (§9). Neither is acceptable; the gate prompt is tuned against
both slices, not just the adversarial one.

---

## 6. Components

| File | Change |
|------|--------|
| `app/normalize.py` | **new** — Arabic normalization, shared by index build and query path. Pure functions. |
| `app/chunking.py` | **new** — split long answers into overlapping chunks, retain parent fatwa id |
| `app/gate.py` | **new** — structured answerability / premise / relevance call |
| `app/citations.py` | **new** — parse `[n]` markers, map to fatwa ids, validate |
| `prepare_data.py` | fix NaN-before-`astype(str)`; parse `categories` → list; normalize; collapse near-duplicates |
| `build_index.py` | `max_length` 512 → 2048; chunk-level; dense **+ sparse**; two-field; fp16 storage |
| `app/retrieval.py` | hybrid scoring + RRF + MMR; keep the lazy-load/idle-unload design intact |
| `app/llm.py` | grounded-citation prompts per verdict; abstention and premise-correction paths |
| `app/api.py` | orchestrate gate → stream; three real modes; emit citations in SSE `meta`; **delete `EXACT_THRESHOLD`** |
| `app/schemas.py` | `+verdict`, `+premise_issue`, `+citations` |
| `frontend/index.html` | clickable `[n]` markers, verdict badge, real anchor links |
| `eval/` | **new** — golden set, adversarial set, derived set, harness |

---

## 7. Data and index design

### 7.1 Two-field embeddings

Fixes the asymmetry in §1.1. Per fatwa store a **question vector**; per answer
chunk store an **answer vector**.

```
score(q, fatwa) = α · sim(q, v_question)
                + (1 − α) · max over chunks of sim(q, v_answer_chunk)
```

α is tuned on the golden set. Question-to-question matching becomes dominant,
which matches what users actually send.

### 7.2 Index size

~18.7k question vectors + ~28k answer-chunk vectors ≈ 47k × 1024. At float32 that
is ~190MB (up from 77MB). **Store as float16** → ~96MB, negligible recall loss.
This matters because the index is committed to git.

### 7.3 Data fixes

- Drop rows whose answer is null **before** `astype(str)` — removes the 72 `"nan"` rows.
- Parse `categories` from `"['التفسير']"` into a real list; expose it through the API.
- Apply `normalize.py` at index time and query time (same function, no drift).
- Collapse near-duplicates at ≥0.99, keeping the longer answer.
- Build the index with the same dtype/precision used at query time (current code
  builds with `use_fp16=True` and queries at fp32).

---

## 8. Error handling

**The gate fails closed.** On API error, timeout, or malformed JSON (one retry
with a stricter instruction, then stop), the system abstains with a service
message. It must never fall back to answering ungrounded — that would silently
reinstate the current bug under load.

| Condition | Behavior |
|-----------|----------|
| Gate API error / timeout | abstain + service message |
| Gate returns malformed JSON | one retry, then abstain |
| Gate cites an id not in the candidate set | drop that id; abstain if none remain |
| Generation emits `[n]` for an uncited fatwa | strip marker, flag answer as unverified |
| `finish_reason == "length"` | explicit truncation notice, not a silently cut ruling |
| Mid-stream failure | SSE `error` + incomplete-answer marker |
| HF rate limit | distinct, actionable message |

---

## 9. Evaluation

Nothing ships without a metric. The harness is built **first** (Phase 0).

### Primary metric — answer faithfulness
LLM-as-judge scores every claim in a generated answer against the fatwa it cites.
Reported as % of claims supported. Set: ~80 questions spanning `direct` and
`derived`.

### Guard metric — false abstention rate
**~60 derived-answerable questions**: paraphrases, dialect, generalizations, and
novel applications where a correct derived answer exists from the corpus.
Correct behavior is `derived` with correct citations, **never** `abstain`.
This is the guard against the design collapsing into over-refusal.

### Safety metric — abstention precision/recall
~60 adversarial questions: false premises (incl. the Hajj/Ramadan case),
out-of-scope, gibberish. Correct behavior is `abstain`.

### Retrieval metrics
~200 golden pairs (question → expected fatwa id). recall@k, MRR, nDCG.
Current baseline to beat: recall@1 = 97.7%, recall@5 = 100%, MRR = 0.988.

Once chunking lands, retrieval returns *chunks*, not fatwas. Metrics are computed
against the **parent fatwa id** — a hit is correct if the chunk's parent is the
expected fatwa, deduplicated by parent before ranking. The baseline above was
measured pre-chunking and remains comparable under this rule.

### Misattribution check
On the `derived` set, assert the answer never presents an applied ruling as a
direct quotation from Ibn Baz. Zero tolerance.

### Unit tests
normalization, chunking, RRF, MMR, citation parsing — pure functions, fast.

### Regression
Snapshot gate verdicts on the adversarial and derived sets; diff on every change.

---

## 10. Phases

| Phase | Content | Exit criterion |
|-------|---------|----------------|
| **0** | Eval harness + all four sets | Baseline measured and committed |
| **1** | Data fixes, normalization, chunking, index rebuild (two-field, dense+sparse, fp16) | Retrieval metrics ≥ baseline; 0 `"nan"` rows; corpus coverage ≥ 98% |
| **2** | Hybrid retrieval + RRF + MMR | Retrieval metrics improve; no duplicate pairs in top-5 |
| **3** | Gate + verdict prompts + citations; delete `EXACT_THRESHOLD` | Faithfulness ↑; false-abstention ≤ target; Hajj/Ramadan abstains |
| **4** | Frontend citations + verdict badge | `[n]` clickable; verdict visible |
| **5** | Hardening: rate limit, CORS fix, input validation, query logging, README/Space repair | — |

Phases 0, 1 and 3 fix the reported bug. Phases 2, 4, 5 are quality and
production concerns.

**Planning granularity.** This spec is too large for one implementation plan.
Each phase gets its own plan, written after the previous phase's exit criterion
is met — so Phase 2's plan can use Phase 1's measured numbers rather than guesses.
The first plan covers **Phase 0 + Phase 1** together, since the harness is
meaningless until there is a rebuilt index to measure and the rebuild is
unverifiable without the harness.

---

## 11. Risks

| Risk | Mitigation |
|------|------------|
| **Gate over-abstains, app becomes useless** | Dedicated false-abstention metric (§9) on a 60-question derived set; gate prompt tuned against it, not only against the adversarial set |
| Gate adds latency | Small structured call (~100 output tokens) before streaming; measure and budget |
| Gate adds per-question API cost | One extra small call; acceptable vs. a local reranker's RAM cost on a 4GB box |
| Index rebuild is a multi-hour CPU run | One-time; ship the rebuilt index in git as today |
| Index grows 77MB → 96MB in git | Accepted; fp16 keeps it bounded |
| `derived` answers drift into misattribution | Structural separation in the prompt + zero-tolerance misattribution check (§9) |
| Chunking splits a ruling from its exception clause | Overlapping chunks; parent-fatwa context available to the gate |

---

## 12. Open items for the implementation plan

- α weighting between question and answer fields — tune on the golden set.
- Chunk size and overlap — start ~400 tokens / 80 overlap, tune.
- Which model serves the gate (may differ from the answer model).
- Golden set construction: sample from the corpus vs. author by hand vs. both.
- Whether the `derived` and adversarial sets need review by someone with shariah
  knowledge before they are treated as ground truth. Labelling a question
  "answerable by derivation" is itself a scholarly judgement, and a wrong label
  trains the gate toward the wrong behavior.
