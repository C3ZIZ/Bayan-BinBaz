---
title: Bayan - Bin Baz Fatwa Assistant (LLM + RAG)
emoji: 📚
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Bayan – BinBaz Fatwa Assistant (Arabic)

Bayan is an Arabic fatwa assistant built on the published fatwas of **Shaykh Abdulaziz ibn Baz (رحمه الله)**.

Its governing rule is that **the system's default is not to assert.** An answer is a
claim about what the shaykh ruled, so every ruling it states must be traceable to a
specific retrieved fatwa — and when the retrieved fatwas do not establish an answer,
it says so instead of improvising one.

> ⚠️ **Religious disclaimer**
> This project is for educational and research purposes. It is **not** an official
> project of Shaykh Ibn Baz, his estate, or any Saudi religious body. Answers are
> machine-generated and **do not replace asking qualified scholars**.

---

## How it works

```text
question
  ↓
normalize        strip tashkeel, fold alef/alef-maqsura, unify digits
  ↓
split            multi-part questions → one retrieval per part, merged
  ↓
retrieve         dense (BGE-M3, two-field) + BM25 → RRF → MMR
  ↓
GATE             one structured LLM call: is this answerable, and from which fatwas?
  ↓
generate         streamed, sees ONLY gate-approved fatwas, cites [n] per ruling
  ↓
UI               [n] renders as a link to the fatwa on binbaz.org.sa
```

### The three verdicts

| verdict | when | behaviour |
|---------|------|-----------|
| `direct` | a fatwa answers **this very question** | answer + cite `[n]` |
| **`derived`** | **no exact fatwa, but retrieved ones establish the ruling** | answer + cite the nearest fatwas, and state plainly that this is an inference, not a direct ruling by the shaykh |
| `abstain` | false premise, out of scope, gibberish, or nothing related in the corpus | explain, cite nothing, assert no ruling |

`derived` is the **common** case, not a fallback. Measured on this corpus, a fatwa's
own question retrieves itself at a median cosine of only **0.815**, so treating
"no exact match" as "no answer" would refuse most legitimate questions. `direct` is
deliberately strict — a dialectal rephrasing, a narrower case, or applying a ruling
to the asker's situation are all `derived`, so the UI never labels an inference as
a matched fatwa.

A **failed** gate call is reported separately from a genuine abstention. Saying
"no fatwa exists" when the service simply could not be reached would be a false
claim about the shaykh's fatwas, so the UI distinguishes
*افتراض غير صحيح* / *لا توجد فتوى* / *الخدمة غير متاحة*.

### LLM backend: `api`, `local`, or `both`

Each question costs **two** LLM calls (gate, then generation), so a depleted
hosted quota takes the whole app down. `LLM_BACKEND` controls this:

| mode | behaviour |
|------|-----------|
| `api` | Hugging Face Inference only — best quality, metered |
| `local` | GGUF model in-process — no quota, no per-question cost, works offline |
| **`both`** *(default)* | **API first, local as fallback** on 402 / 429 / timeout / outage |

The fallback only engages **before the first token reaches the client** — text
already streamed cannot be retracted, and continuing from a different model would
splice two answers together.

**The local model runs the gate but never writes a ruling.** Measured on the real
app, the 3B model answered «هل اقدر اصلي العشاء ٥ ركعات؟» with *"permissible"* — a
fabricated ruling on an obligatory prayer, reached by misapplying a fatwa about
*voluntary* prayer, and carrying **valid citations**. The citation validator cannot
catch that: the markers are real, the inference is false. So when the hosted model
is unavailable the app says so and lists the retrieved fatwas with links.
`LOCAL_ALLOW_RULINGS=1` overrides this.

Speed, measured in Docker on CPU: **~1–4 tok/s**, with prompt prefill dominating.
(An earlier 22 tok/s figure came from a native Metal-accelerated build on macOS and
does not apply to a CPU deployment.)

The local default is **Falcon-H1-3B-Instruct**, chosen by benchmarking the two
jobs this app actually does rather than by leaderboard:

| model | gate correct | Arabic | cites | speed | size |
|-------|--------------|--------|-------|-------|------|
| **Falcon-H1-3B (TII)** | **3/3** | 100% | ✅ | 22 tok/s* | 1.9 GB |
| Qwen2.5-3B | 2/3 | 100% | ✅ | 27 tok/s* | 2.1 GB |
| ALLaM-7B (SDAIA) | 2/3 | 100% | ✅ | 12 tok/s* | 4.3 GB |

\* native macOS/Metal. On CPU in Docker expect ~1–4 tok/s.

ALLaM is Arabic-native and twice the size, yet scored no better on the gate, ran
at half the speed, and echoed source blocks instead of answering — bigger and
Arabic-specific did not win here. Reproduce with
`python eval/bench_local_models.py`.

Small models over-refuse without worked examples: the 3B model initially scored
2/4 end-to-end, refusing a valid dialectal question. Three few-shot examples in
the gate prompt take it to **5/5**. Verify any local model before trusting it:

```bash
docker compose -f docker-compose.test.yml run --rm \
  -e LLM_BACKEND=local tests python eval/check_gate_cases.py
```

### Why a gate rather than a similarity threshold

Measured against the live index, the similarity distributions of answerable and
unanswerable questions **overlap**, so no threshold can separate them:

| query | top-1 similarity |
|-------|------------------|
| «إذا جا الحج ورمضان في نفس الوقت…» (impossible — Ramadan is month 9, Hajj months are 10–12) | **0.628** |
| «ما حكم صيام يوم عرفة لغير الحاج؟» (valid) | 0.673 |
| «ودي اعرف حكم الاغاني وش رايك» (valid, dialect) | **0.499** |
| «كيف أطبخ الكبسة السعودية؟» (out of scope) | 0.473 |
| gibberish | 0.450 |

The in-scope minimum (0.499) sits *below* the bad-query maximum (0.628). The decision
therefore has to be made by something that **reads** the candidates, not one that ranks
them. The gate **fails closed**: on API error, timeout, or unparseable output it
abstains rather than falling back to answering ungrounded.

---

## Data

- **Source:** the official site <https://binbaz.org.sa/> — always the authority for the full text.
- **Dataset:** [Bin Baz Fatwas Dataset – Kaggle](https://www.kaggle.com/datasets/a5medashraf/bin-baz-fatwas-dataset) (belongs to its original author; check the Kaggle page for licensing).
- **After cleaning:** 18,681 fatwas, 215 categories, 100% link coverage.

`prepare_data.py` fixes three defects in the original pipeline: `.astype(str)` ran
before `dropna` (so 72 rows reached the index with the literal string `"nan"` as their
answer), `categories` was never parsed out of its stringified list, and duplicate links
were never collapsed.

## Index

`build_index.py` writes into `data/index/`:

| artifact | contents |
|----------|----------|
| `question_emb.npy` | `(F, 1024) float16` — one vector per fatwa question |
| `answer_emb.npy` | `(C, 1024) float16` — one vector per answer chunk |
| `answer_parent.npy` | `(C,) int32` — chunk → fatwa row |
| `fatwas_meta.parquet` | id, question, title, answer, link, categories |
| `index_manifest.json` | normalize version, model, chunk settings, counts |

Two changes drive retrieval quality:

- **Chunking.** The old index encoded at `max_length=512` while BGE-M3 supports 8192.
  Measured with the real tokenizer that truncated **17.7% of fatwas** and left
  **19.6% of all corpus tokens** out of the index entirely.
- **Two fields.** The old index embedded `question + answer` as one vector, of which
  **81% of tokens were answer text** — while users send short questions. Questions and
  answer chunks are now embedded separately and blended at query time, weighted by
  `RETRIEVAL_ALPHA`. A fatwa is scored by its *best* matching chunk, not its average.

Vectors are L2-normalized and stored `float16`, since the index is committed to git.

---

## Quickstart

```bash
git clone https://github.com/C3ZIZ/Bayan-BinBaz.git
cd Bayan-BinBaz
cp .env.example .env      # set HF_TOKEN (optional — see LLM_BACKEND)
docker compose up -d --build
```

Open `http://localhost:8000/`. Health: `curl http://localhost:8000/health`.

The prebuilt index ships in the repo, so `prepare_data.py` / `build_index.py` are only
needed to rebuild it.

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Tests

The suite runs in Docker against the same Python 3.11 the app uses:

```bash
docker compose -f docker-compose.test.yml run --rm tests pytest -v
```

### Evaluation

Nothing in the retrieval stack should change without running this:

```bash
docker compose -f docker-compose.test.yml run --rm tests python eval/build_datasets.py
docker compose -f docker-compose.test.yml run --rm tests python eval/run_retrieval_eval.py --tag mychange
```

| dataset | size | what it guards |
|---------|------|----------------|
| `golden.jsonl` | 200 | retrieval accuracy (recall@k, MRR, nDCG) |
| `derived.jsonl` | 60 | **false abstention** — questions with no exact match that must still be answered |
| `adversarial.jsonl` | 60 | **safety** — false premises, out of scope, gibberish; must abstain |

The derived set is the guard against the system collapsing into over-refusal. Its rows
were anchored to real fatwas by lexical keyword search — deliberately *not* by the dense
retriever, which would make the retrieval check circular.

### Deploy on Coolify

Coolify routes through its own proxy, so **do not publish a host port**. The base
`docker-compose.yml` only `expose`s 8000; `docker-compose.override.yml` (loaded
automatically by local `docker compose`, ignored by Coolify) publishes it.

1. Create the resource from this Git repo.
2. Add `HF_TOKEN` under **Environment Variables**.
3. Set a **Domain** on the `bayan` service and deploy.

### Memory budget (~4GB VPS)

| piece | where | RAM |
|-------|-------|-----|
| Index (NumPy) + BM25 + web app | local, always | ~0.4–0.6 GB |
| **BGE-M3** query embedder | local, **lazy-loaded** | ~2.3 GB *while active* |
| LLM (gate + generation) | **Hugging Face Inference API** | 0 GB local |

After `MODEL_IDLE_TIMEOUT` seconds of no traffic, BGE-M3 is unloaded and the memory
returned to the OS via `malloc_trim`. It reloads on the next question.

---

## API

- `GET  /health` — liveness (does not load models)
- `POST /api/chat` — `{"question": "…", "top_k": 5}` → verdict, answer, citations, related fatwas
- `POST /api/chat/stream` — same input as SSE: `meta` → `token`… → `done` (carries resolved citations)

Both are rate limited; exceeding it returns `429` with `Retry-After`.

---

## Environment variables

Nothing is strictly required — the default `LLM_BACKEND=both` falls back to a local
model when the hosted API is unavailable, so the app runs without `HF_TOKEN`.
Set `HF_TOKEN` anyway: answers are far better from the hosted model.

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_BACKEND` | **`both`** | `api` \| `local` \| `both`. Default falls back to the local model when the API fails |
| `LOCAL_LLM_REPO` | `tiiuae/Falcon-H1-3B-Instruct-GGUF` | Local GGUF repo. Weight downloads are free — only Inference is metered |
| `LOCAL_LLM_FILE` | `Falcon-H1-3B-Instruct-Q4_K_M.gguf` | Quant to load (~1.9GB) |
| `LOCAL_LLM_PATH` | — | Point at a `.gguf` on disk to skip the download |
| `LOCAL_LLM_CTX` | `4096` | Local context window |
| `LOCAL_UNLOAD_EMBEDDER` | `0` | Set `1` on a ~4GB host: the local LLM and BGE-M3 don't both fit |
| `RETRIEVAL_HYBRID` | `1` | Fuse BM25 with dense via RRF |
| `RETRIEVAL_MMR` | `1` | Diversify top-k (1.5% of the corpus has a ≥0.99 twin) |
| `RATE_LIMIT_REQUESTS` | `20` | Per window, per client |
| `TRUST_PROXY_HEADER` | `0` | Honour `X-Forwarded-For`. Only behind a trusted proxy — otherwise the limit is bypassable |
| `CORS_ORIGINS` | `*` | Comma-separated allowlist |
| `QUERY_LOG_PATH` | — | JSONL query log (no client identity). Grows the eval sets from real traffic |
| `HF_TOKEN` | — *(recommended)* | <https://huggingface.co/settings/tokens> → Read, with Inference Providers enabled |
| `LLM_API_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | Answer model. Arabic-strong alternatives: Llama-3.3-70B, aya-expanse-32b |
| `GATE_MODEL` | = `LLM_API_MODEL` | Gate can use a cheaper/faster model |
| `LLM_PROVIDER` | `auto` | HF Inference Providers routing |
| `LLM_MAX_TOKENS` | `900` | Arabic is token-hungry; short budgets truncate rulings |
| `LLM_TEMPERATURE` | `0.1` | Keep low for faithfulness |
| `LLM_CONTEXT_CHARS` | `1400` | Chars of each fatwa in the prompt, cut at a **sentence boundary** |
| `GATE_MAX_TOKENS` | `400` | Gate output budget |
| `GATE_TEMPERATURE` | `0.0` | Deterministic verdicts |
| `GATE_SNIPPET_CHARS` | `600` | Chars of each candidate shown to the gate |
| `EMB_MODEL_NAME` | `BAAI/bge-m3` | Must match the model the index was built with |
| `EMB_USE_FP16` | `0` | Keep `0` on CPU |
| `RETRIEVAL_ALPHA` | `0.65` | Weight on question-vs-answer similarity |
| `RETRIEVAL_HYBRID` | `1` | Fuse BM25 with dense via RRF |
| `RETRIEVAL_MMR` | `1` | Diversify top-k (1.5% of the corpus has a ≥0.99 twin) |
| `RETRIEVAL_DEPTH` | `50` | Candidates fused before diversification |
| `RETRIEVAL_MMR_LAMBDA` | `0.7` | 1.0 = pure relevance, lower = more diverse |
| `RATE_LIMIT_REQUESTS` | `20` | Requests per window per client |
| `RATE_LIMIT_WINDOW` | `60` | Window in seconds |
| `CORS_ORIGINS` | `*` | Comma-separated allowlist |
| `MODEL_IDLE_TIMEOUT` | `600` | Seconds before BGE-M3 is unloaded; `0` = never |
| `MODEL_IDLE_CHECK_INTERVAL` | `60` | Idle monitor interval |
| `PORT` | `8000` | Host port (container always listens on 8000) |
| `OMP_NUM_THREADS` | `4` | Set to your vCPU count |

---

## Notes

- **Data is committed on purpose** (`data/raw`, `data/processed`, `data/index`) so the app ships ready-to-run.
- The LLM runs entirely via the **Hugging Face Inference API**; the previous local `llama-cpp` path is kept for reference in `app/archive/llm.py`.
- Design and implementation notes live in [`docs/superpowers/`](docs/superpowers/).
