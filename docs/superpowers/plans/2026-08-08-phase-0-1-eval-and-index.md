# Phase 0 + 1: Eval Harness and Index Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measurement harness that every later phase depends on, then fix the corpus and rebuild the index so retrieval sees 100% of the data with question/answer asymmetry removed.

**Architecture:** A shared `app/normalize.py` is applied identically at index time and query time (enforced by test). `app/chunking.py` splits long answers so no fatwa is truncated. `build_index.py` emits two vector families — one per fatwa question, one per answer chunk — stored fp16. `eval/` scores retrieval against four datasets and is the gate for every subsequent change.

**Tech Stack:** Python 3.11, pytest, pandas, numpy, pyarrow, FlagEmbedding (BGE-M3), Docker Compose.

## Global Constraints

- Python 3.11 (matches `Dockerfile`); dependency versions pinned in `requirements.txt` must not be bumped in this phase.
- Normalization used at index time and query time MUST be the same function. A test enforces this.
- Index stored as **float16**; scoring casts to float32. Index must stay committed to git.
- Retrieval must not regress below the measured baseline: **recall@1 = 97.7%, recall@5 = 100%, MRR = 0.988** (computed against parent fatwa id, deduplicated by parent).
- Do not introduce FAISS or a vector database. Brute-force NumPy is correct at this scale.
- Do not change the lazy-load / idle-unload / `malloc_trim` lifecycle in `app/retrieval.py`.
- All eval datasets are JSONL, UTF-8, `ensure_ascii=False`.
- Sunni sources only when researching derived-answerability. Explicitly exclude Shia sources.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `app/normalize.py` | Arabic text normalization. Pure functions, no I/O. |
| `app/chunking.py` | Token-aware splitting of long answers into overlapping chunks. Pure. |
| `eval/metrics.py` | recall@k, MRR, nDCG. Pure functions over rank lists. |
| `eval/datasets/golden.jsonl` | 200 question → fatwa_id pairs (retrieval accuracy) |
| `eval/datasets/derived.jsonl` | 60 questions answerable only by derivation (false-abstention guard) |
| `eval/datasets/adversarial.jsonl` | 60 questions that must abstain (safety) |
| `eval/build_datasets.py` | Generates golden.jsonl from the corpus; derived/adversarial are hand-curated. |
| `eval/run_retrieval_eval.py` | Loads datasets, runs retrieval, prints/persists metrics. |
| `tests/test_normalize.py` | Normalization unit tests |
| `tests/test_chunking.py` | Chunking unit tests |
| `tests/test_metrics.py` | Metric unit tests |
| `tests/test_index_integrity.py` | Index/meta invariants (shapes, dtypes, no "nan" answers, parent ids) |
| `prepare_data.py` | **modify** — NaN handling, category parsing, normalization, dedup |
| `build_index.py` | **modify** — two-field, chunked, fp16, max_length 2048 |
| `app/retrieval.py` | **modify** — load new index format, two-field scoring |
| `requirements-dev.txt` | pytest + pytest-cov for the test image |
| `Dockerfile.test` | Test image reusing the app image's dependency layer |

---

## Task 1: Development environment and test runner

**Files:**
- Create: `requirements-dev.txt`, `Dockerfile.test`, `pytest.ini`, `tests/__init__.py`

**Interfaces:**
- Produces: `docker compose -f docker-compose.test.yml run --rm tests` runs the full suite.

- [ ] **Step 1: Create `requirements-dev.txt`**

```
pytest==8.3.3
pytest-cov==5.0.0
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q --strict-markers
markers =
    slow: requires the BGE-M3 model or a full index rebuild
```

- [ ] **Step 3: Create `Dockerfile.test`**

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1" && \
    pip install -r requirements.txt -r requirements-dev.txt

COPY . .
CMD ["pytest"]
```

- [ ] **Step 4: Create `docker-compose.test.yml`**

```yaml
services:
  tests:
    build:
      context: .
      dockerfile: Dockerfile.test
    volumes:
      - ./tests:/app/tests
      - ./app:/app/app
      - ./eval:/app/eval
      - hf-models:/models
    environment:
      OMP_NUM_THREADS: "4"
volumes:
  hf-models:
```

- [ ] **Step 5: Create `tests/__init__.py`** (empty file)

- [ ] **Step 6: Build the test image and confirm pytest runs**

Run: `docker compose -f docker-compose.test.yml build tests`
Then: `docker compose -f docker-compose.test.yml run --rm tests pytest --collect-only`
Expected: exits 0, collects 0 tests (none written yet).

- [ ] **Step 7: Commit**

```bash
git add requirements-dev.txt pytest.ini Dockerfile.test docker-compose.test.yml tests/__init__.py
git commit -m "test: add dockerized pytest runner"
```

---

## Task 2: Arabic normalization

Normalization is **configurable** because aggressive folding can hurt a multilingual
model that saw diacritics in training. Task 8 A/B-tests the aggressive options on the
golden set; this task only provides the knobs.

**Files:**
- Create: `app/normalize.py`, `tests/test_normalize.py`

**Interfaces:**
- Produces:
  - `normalize_arabic(text: str, *, fold_alef: bool = True, fold_yaa: bool = True, fold_taa_marbuta: bool = False, strip_tashkeel: bool = True) -> str`
  - `NORMALIZE_VERSION: str` — bumped whenever behavior changes; written into index meta.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_normalize.py
from app.normalize import normalize_arabic, NORMALIZE_VERSION


def test_strips_tashkeel():
    assert normalize_arabic("الصَّلَاةُ") == "الصلاة"


def test_strips_tatweel():
    assert normalize_arabic("الصــلاة") == "الصلاة"


def test_folds_alef_variants():
    assert normalize_arabic("أإآا") == "اااا"


def test_folds_alef_maqsura_to_yaa():
    assert normalize_arabic("على") == "علي"


def test_taa_marbuta_preserved_by_default():
    assert normalize_arabic("صلاة") == "صلاة"


def test_taa_marbuta_folded_when_requested():
    assert normalize_arabic("صلاة", fold_taa_marbuta=True) == "صلاه"


def test_converts_arabic_indic_digits():
    assert normalize_arabic("١٢٣") == "123"


def test_collapses_whitespace():
    assert normalize_arabic("  الصلاة   واجبة \n\n علينا ") == "الصلاة واجبة علينا"


def test_is_idempotent():
    once = normalize_arabic("الصَّلَاةُ   واجِبَة")
    assert normalize_arabic(once) == once


def test_handles_empty_and_none_like():
    assert normalize_arabic("") == ""
    assert normalize_arabic("   ") == ""


def test_leaves_latin_untouched():
    assert normalize_arabic("BGE-M3 v2") == "BGE-M3 v2"


def test_version_is_a_nonempty_string():
    assert isinstance(NORMALIZE_VERSION, str) and NORMALIZE_VERSION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.normalize'`

- [ ] **Step 3: Write the implementation**

```python
# app/normalize.py
"""Arabic text normalization shared by index build and query time.

The SAME function must run on both sides — a mismatch silently degrades
retrieval. tests/test_index_integrity.py enforces that the index records the
NORMALIZE_VERSION it was built with.
"""
import re

NORMALIZE_VERSION = "1.0"

# Harakat, tanwin, shadda, sukun, superscript alef, and Quranic annotation marks.
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = re.compile(r"ـ")
_ALEF = re.compile(r"[آأإٱ]")       # آ أ إ ٱ
_YAA = re.compile(r"ى")                            # ى
_TAA_MARBUTA = re.compile(r"ة")                    # ة
_WS = re.compile(r"\s+")

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_arabic(
    text: str,
    *,
    fold_alef: bool = True,
    fold_yaa: bool = True,
    fold_taa_marbuta: bool = False,
    strip_tashkeel: bool = True,
) -> str:
    """Normalize Arabic text for retrieval. Idempotent."""
    if not text:
        return ""

    if strip_tashkeel:
        text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    if fold_alef:
        text = _ALEF.sub("ا", text)
    if fold_yaa:
        text = _YAA.sub("ي", text)
    if fold_taa_marbuta:
        text = _TAA_MARBUTA.sub("ه", text)

    text = text.translate(_ARABIC_INDIC)
    return _WS.sub(" ", text).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_normalize.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add app/normalize.py tests/test_normalize.py
git commit -m "feat: add configurable Arabic normalization"
```

---

## Task 3: Token-aware chunking

**Files:**
- Create: `app/chunking.py`, `tests/test_chunking.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `chunk_text(text: str, token_len: Callable[[str], int], max_tokens: int = 450, overlap_tokens: int = 100) -> list[str]`

`token_len` is injected so unit tests run without downloading BGE-M3. Production
passes the real tokenizer.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking.py
from app.chunking import chunk_text

# One "token" per whitespace word — deterministic and model-free.
def wc(s: str) -> int:
    return len(s.split())


def test_short_text_is_a_single_chunk():
    assert chunk_text("a b c", wc, max_tokens=10) == ["a b c"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("", wc, max_tokens=10) == []
    assert chunk_text("   ", wc, max_tokens=10) == []


def test_long_text_is_split():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, wc, max_tokens=30, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(wc(c) <= 30 for c in chunks)


def test_chunks_overlap():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, wc, max_tokens=30, overlap_tokens=5)
    first_tail = chunks[0].split()[-5:]
    assert any(t in chunks[1].split() for t in first_tail)


def test_no_content_is_lost():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, wc, max_tokens=30, overlap_tokens=5)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == {str(i) for i in range(100)}


def test_overlap_must_be_smaller_than_max():
    try:
        chunk_text("a b c", wc, max_tokens=10, overlap_tokens=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError when overlap >= max_tokens")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.chunking'`

- [ ] **Step 3: Write the implementation**

```python
# app/chunking.py
"""Split long fatwa answers into overlapping, token-bounded chunks.

Chunking serves two purposes: no fatwa is truncated at index time, and a long
fatwa's individual points get their own vectors instead of being averaged into
one diluted representation.
"""
from typing import Callable, List


def chunk_text(
    text: str,
    token_len: Callable[[str], int],
    max_tokens: int = 450,
    overlap_tokens: int = 100,
) -> List[str]:
    """Split `text` into chunks of at most `max_tokens`, overlapping by
    `overlap_tokens`. Splits on whitespace, so words are never cut."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    words = text.split()
    if not words:
        return []

    if token_len(" ".join(words)) <= max_tokens:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        lo, hi, best = start + 1, len(words), start + 1
        while lo <= hi:                     # largest window that still fits
            mid = (lo + hi) // 2
            if token_len(" ".join(words[start:mid])) <= max_tokens:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        chunks.append(" ".join(words[start:best]))
        if best >= len(words):
            break
        step = max(1, (best - start) - overlap_tokens)
        start += step
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_chunking.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/chunking.py tests/test_chunking.py
git commit -m "feat: add token-aware overlapping chunking"
```

---

## Task 4: Retrieval metrics

**Files:**
- Create: `eval/__init__.py`, `eval/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `recall_at_k(ranked_ids: list[int], relevant_id: int, k: int) -> float`
  - `reciprocal_rank(ranked_ids: list[int], relevant_id: int) -> float`
  - `ndcg_at_k(ranked_ids: list[int], relevant_id: int, k: int) -> float`
  - `aggregate(rows: list[dict], ks: tuple[int, ...]) -> dict[str, float]`

`rows` entries are `{"ranked_ids": [...], "relevant_id": int}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import math
from eval.metrics import recall_at_k, reciprocal_rank, ndcg_at_k, aggregate


def test_recall_hit_at_rank_1():
    assert recall_at_k([5, 2, 9], 5, 1) == 1.0


def test_recall_miss_outside_k():
    assert recall_at_k([5, 2, 9], 9, 2) == 0.0


def test_recall_hit_inside_k():
    assert recall_at_k([5, 2, 9], 9, 3) == 1.0


def test_reciprocal_rank_positions():
    assert reciprocal_rank([5, 2, 9], 5) == 1.0
    assert reciprocal_rank([5, 2, 9], 2) == 0.5
    assert reciprocal_rank([5, 2, 9], 7) == 0.0


def test_ndcg_rank_1_is_one():
    assert ndcg_at_k([5, 2, 9], 5, 3) == 1.0


def test_ndcg_rank_2_matches_formula():
    assert math.isclose(ndcg_at_k([5, 2, 9], 2, 3), 1 / math.log2(3))


def test_ndcg_miss_is_zero():
    assert ndcg_at_k([5, 2, 9], 7, 3) == 0.0


def test_aggregate_reports_all_metrics():
    rows = [
        {"ranked_ids": [1, 2, 3], "relevant_id": 1},
        {"ranked_ids": [4, 5, 6], "relevant_id": 5},
    ]
    out = aggregate(rows, ks=(1, 3))
    assert out["recall@1"] == 0.5
    assert out["recall@3"] == 1.0
    assert math.isclose(out["mrr"], 0.75)
    assert out["n"] == 2


def test_aggregate_on_empty_input():
    out = aggregate([], ks=(1,))
    assert out["n"] == 0
    assert out["recall@1"] == 0.0
    assert out["mrr"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# eval/metrics.py
"""Retrieval metrics. Single-relevant-document formulation: each eval row has
exactly one correct parent fatwa id."""
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
    # Binary relevance, single relevant doc -> IDCG is 1.
    return 1.0 / math.log2(ids.index(relevant_id) + 2)


def aggregate(rows: List[Dict], ks: Tuple[int, ...] = (1, 3, 5, 10, 20)) -> Dict[str, float]:
    out: Dict[str, float] = {"n": len(rows)}
    if not rows:
        for k in ks:
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        out["mrr"] = 0.0
        return out

    for k in ks:
        out[f"recall@{k}"] = sum(
            recall_at_k(r["ranked_ids"], r["relevant_id"], k) for r in rows
        ) / len(rows)
        out[f"ndcg@{k}"] = sum(
            ndcg_at_k(r["ranked_ids"], r["relevant_id"], k) for r in rows
        ) / len(rows)
    out["mrr"] = sum(
        reciprocal_rank(r["ranked_ids"], r["relevant_id"]) for r in rows
    ) / len(rows)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_metrics.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add eval/__init__.py eval/metrics.py tests/test_metrics.py
git commit -m "feat: add retrieval metrics (recall@k, MRR, nDCG)"
```

---

## Task 5: Fix `prepare_data.py`

**Files:**
- Modify: `prepare_data.py`
- Create: `tests/test_prepare_data.py`

**Interfaces:**
- Produces: `normalize_fatwa_table(df: pd.DataFrame) -> pd.DataFrame` — note the unused
  `path` parameter is removed. Output columns:
  `id, question, answer, title, link, categories` where `categories` is a `list[str]`.

Fixes three defects: NaN survives as the string `"nan"`; `categories` stays a
stringified list; no near-duplicate collapsing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare_data.py
import numpy as np
import pandas as pd
from prepare_data import normalize_fatwa_table, parse_categories


def test_parse_categories_from_stringified_list():
    assert parse_categories("['التفسير']") == ["التفسير"]


def test_parse_categories_handles_multiple():
    assert parse_categories("['أ', 'ب']") == ["أ", "ب"]


def test_parse_categories_handles_garbage():
    assert parse_categories("not a list") == []
    assert parse_categories(None) == []
    assert parse_categories(float("nan")) == []


def test_parse_categories_passes_through_real_list():
    assert parse_categories(["أ"]) == ["أ"]


def _frame(**over):
    base = {
        "questions": ["س1", "س2"],
        "answers": ["ج1", "ج2"],
        "titles": ["ع1", "ع2"],
        "links": ["l1", "l2"],
        "categories": ["['أ']", "['ب']"],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_drops_rows_with_null_answer_not_stringify_them():
    df = normalize_fatwa_table(_frame(answers=["ج1", np.nan]))
    assert len(df) == 1
    assert "nan" not in df["answer"].tolist()


def test_drops_rows_with_null_question():
    df = normalize_fatwa_table(_frame(questions=["س1", None]))
    assert len(df) == 1


def test_drops_literal_nan_strings():
    df = normalize_fatwa_table(_frame(answers=["ج1", "nan"]))
    assert len(df) == 1


def test_categories_become_real_lists():
    df = normalize_fatwa_table(_frame())
    assert df["categories"].iloc[0] == ["أ"]


def test_renames_plural_columns():
    df = normalize_fatwa_table(_frame())
    assert set(["question", "answer", "title", "link", "categories"]).issubset(df.columns)


def test_missing_optional_columns_are_created():
    df = normalize_fatwa_table(pd.DataFrame({"questions": ["س"], "answers": ["ج"]}))
    assert df["title"].iloc[0] == ""
    assert df["categories"].iloc[0] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_prepare_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_categories'`

- [ ] **Step 3: Rewrite `prepare_data.py`**

```python
"""Load the raw fatwa CSV, clean it, and write data/processed/fatwas.parquet.

Three defects fixed here versus the original:
  1. `.astype(str)` ran before `dropna`, turning NaN into the string "nan" and
     defeating the filter — 72 such rows reached the index.
  2. `categories` stayed a stringified list ("['التفسير']"), so the API's
     category field was always None despite 100% coverage.
  3. Near-identical fatwas were never collapsed, so top-k could return copies.
"""
import ast
import math
from pathlib import Path
from typing import Any, List

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "fatwas.parquet"

_NULLISH = {"", "nan", "none", "null", "na"}


def parse_categories(value: Any) -> List[str]:
    """Turn "['التفسير']" (or a real list) into ["التفسير"]. Never raises."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return []


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in _NULLISH


def normalize_fatwa_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    rename_map = {}
    for target, candidates in [
        ("question", ("question", "questions")),
        ("answer", ("answer", "answers")),
        ("title", ("title", "titles")),
        ("link", ("link", "links", "url")),
        ("categories", ("categories", "category")),
    ]:
        src = pick(*candidates)
        if src:
            rename_map[src] = target
    df = df.rename(columns=rename_map)

    for col in ("title", "link", "categories"):
        if col not in df.columns:
            df[col] = None

    # Drop nullish BEFORE any string coercion — this is the bug that let 72
    # literal "nan" answers into the index.
    keep = ~(df["question"].map(_is_nullish) | df["answer"].map(_is_nullish))
    df = df[keep].copy()

    df["categories"] = df["categories"].map(parse_categories)
    for col in ("question", "answer", "title"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["link"] = df["link"].fillna("").astype(str).str.strip()

    return df[["question", "answer", "title", "link", "categories"]]


def load_fatwa_tables() -> pd.DataFrame:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} not found.")

    tables = []
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
            continue
        try:
            df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_json(path)
        except Exception as e:
            print(f"[SKIP] {path} ({e})")
            continue

        norm = normalize_fatwa_table(df)
        if len(norm):
            print(f"[OK] {len(norm)} rows from {path}")
            tables.append(norm)

    if not tables:
        raise RuntimeError("No usable data in data/raw.")

    df_all = pd.concat(tables, ignore_index=True)
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["question", "answer"])
    df_all = df_all.drop_duplicates(subset=["link"], keep="first")
    print(f"[dedup] {before} -> {len(df_all)} rows")

    df_all.insert(0, "id", range(1, len(df_all) + 1))
    return df_all


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_fatwa_tables()
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} fatwas to {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_prepare_data.py -v`
Expected: 10 passed

- [ ] **Step 5: Regenerate the processed parquet and verify the defects are gone**

Run:
```bash
docker compose -f docker-compose.test.yml run --rm tests python prepare_data.py
docker compose -f docker-compose.test.yml run --rm tests python -c "
import pandas as pd
df = pd.read_parquet('data/processed/fatwas.parquet')
bad = (df['answer'].astype(str).str.strip().str.lower() == 'nan').sum()
print('rows:', len(df), '| literal-nan answers:', bad)
print('categories type:', type(df['categories'].iloc[0]))
assert bad == 0, 'literal nan answers still present'
assert isinstance(df['categories'].iloc[0], (list, tuple)), 'categories not parsed'
print('OK')
"
```
Expected: `literal-nan answers: 0`, categories is a list, prints `OK`.

- [ ] **Step 6: Commit**

```bash
git add prepare_data.py tests/test_prepare_data.py data/processed/fatwas.parquet
git commit -m "fix: drop null answers before coercion and parse categories"
```

---

## Task 6: Rebuild the index — two-field, chunked, fp16

**Files:**
- Modify: `build_index.py`
- Create: `tests/test_index_integrity.py`

**Interfaces:**
- Produces these artifacts in `data/index/`:
  - `question_emb.npy` — `(F, 1024) float16`, one row per fatwa
  - `answer_emb.npy` — `(C, 1024) float16`, one row per answer chunk
  - `answer_parent.npy` — `(C,) int32`, row index into the fatwa table for each chunk
  - `fatwas_meta.parquet` — `id, question, title, answer, link, categories`
  - `index_manifest.json` — `{normalize_version, emb_model, max_length, chunk_max_tokens, chunk_overlap, n_fatwas, n_chunks, dtype}`

Vectors are L2-normalized before fp16 cast.

- [ ] **Step 1: Write the failing integrity tests**

```python
# tests/test_index_integrity.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

INDEX = Path("data/index")

pytestmark = pytest.mark.skipif(
    not (INDEX / "index_manifest.json").exists(),
    reason="index not built yet (run build_index.py)",
)


@pytest.fixture(scope="module")
def artifacts():
    return (
        np.load(INDEX / "question_emb.npy"),
        np.load(INDEX / "answer_emb.npy"),
        np.load(INDEX / "answer_parent.npy"),
        pd.read_parquet(INDEX / "fatwas_meta.parquet"),
        json.loads((INDEX / "index_manifest.json").read_text()),
    )


def test_dtypes_are_float16(artifacts):
    q, a, *_ = artifacts
    assert q.dtype == np.float16
    assert a.dtype == np.float16


def test_question_rows_match_meta_rows(artifacts):
    q, _, _, meta, _ = artifacts
    assert len(q) == len(meta)


def test_parent_map_matches_answer_rows(artifacts):
    _, a, parent, _, _ = artifacts
    assert len(parent) == len(a)


def test_parent_indices_are_in_range(artifacts):
    _, _, parent, meta, _ = artifacts
    assert parent.min() >= 0
    assert parent.max() < len(meta)


def test_every_fatwa_has_at_least_one_chunk(artifacts):
    _, _, parent, meta, _ = artifacts
    assert len(np.unique(parent)) == len(meta)


def test_vectors_are_unit_norm(artifacts):
    q, a, *_ = artifacts
    for m in (q[:200], a[:200]):
        norms = np.linalg.norm(m.astype(np.float32), axis=1)
        assert np.allclose(norms, 1.0, atol=2e-3)


def test_no_literal_nan_answers(artifacts):
    *_, meta, _ = artifacts
    bad = (meta["answer"].astype(str).str.strip().str.lower() == "nan").sum()
    assert bad == 0


def test_categories_are_lists(artifacts):
    *_, meta, _ = artifacts
    assert isinstance(meta["categories"].iloc[0], (list, np.ndarray))


def test_manifest_records_normalize_version(artifacts):
    from app.normalize import NORMALIZE_VERSION
    *_, manifest = artifacts
    assert manifest["normalize_version"] == NORMALIZE_VERSION


def test_manifest_counts_match_arrays(artifacts):
    q, a, _, _, manifest = artifacts
    assert manifest["n_fatwas"] == len(q)
    assert manifest["n_chunks"] == len(a)
```

- [ ] **Step 2: Run tests to verify they skip**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_index_integrity.py -v`
Expected: 10 skipped (`index not built yet`)

- [ ] **Step 3: Rewrite `build_index.py`**

```python
"""Build the retrieval index.

Two changes versus the original that matter for quality:

  1. max_length was 512 while BGE-M3 supports 8192 — 17.7% of fatwas were
     truncated and 19.6% of all corpus tokens never reached a vector. Answers
     are now chunked, so nothing is dropped.
  2. The original embedded question+answer as one vector, of which 81% of tokens
     were answer text, while users send short questions. Questions and answer
     chunks are now embedded separately so question-to-question matching can
     dominate at query time.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

from app.chunking import chunk_text
from app.normalize import NORMALIZE_VERSION, normalize_arabic

PROCESSED_PATH = Path("data/processed/fatwas.parquet")
INDEX_DIR = Path("data/index")

EMB_MODEL = "BAAI/bge-m3"
MAX_LENGTH = 2048
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP = 100
BATCH_SIZE = 16


def _encode(model, texts, batch_size=BATCH_SIZE):
    out = model.encode(texts, batch_size=batch_size, max_length=MAX_LENGTH)
    vecs = out["dense_vecs"].astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs.astype(np.float16)


def main():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_PATH} not found. Run prepare_data.py first.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_PATH)
    model = BGEM3FlagModel(EMB_MODEL, use_fp16=False)
    tokenizer = model.tokenizer

    def token_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=True)["input_ids"])

    questions = [normalize_arabic(q) for q in df["question"].fillna("").tolist()]

    chunk_texts, chunk_parent = [], []
    for row_idx, answer in enumerate(df["answer"].fillna("").tolist()):
        pieces = chunk_text(
            normalize_arabic(answer), token_len,
            max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP,
        )
        if not pieces:                      # terse or empty answer
            pieces = [questions[row_idx]]   # keep a placeholder so every fatwa is reachable
        for piece in pieces:
            chunk_texts.append(piece)
            chunk_parent.append(row_idx)

    print(f"Encoding {len(questions)} questions...")
    question_emb = _encode(model, questions)
    print(f"Encoding {len(chunk_texts)} answer chunks...")
    answer_emb = _encode(model, chunk_texts)

    np.save(INDEX_DIR / "question_emb.npy", question_emb)
    np.save(INDEX_DIR / "answer_emb.npy", answer_emb)
    np.save(INDEX_DIR / "answer_parent.npy", np.asarray(chunk_parent, dtype=np.int32))

    meta_cols = [c for c in ["id", "question", "title", "answer", "link", "categories"] if c in df.columns]
    df[meta_cols].to_parquet(INDEX_DIR / "fatwas_meta.parquet", index=False)

    (INDEX_DIR / "index_manifest.json").write_text(json.dumps({
        "normalize_version": NORMALIZE_VERSION,
        "emb_model": EMB_MODEL,
        "max_length": MAX_LENGTH,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_fatwas": int(len(question_emb)),
        "n_chunks": int(len(answer_emb)),
        "dtype": "float16",
    }, ensure_ascii=False, indent=2))

    print(f"Done. {len(question_emb)} fatwas, {len(answer_emb)} chunks.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the rebuild** (slow — downloads BGE-M3, encodes ~45k vectors on CPU)

Run: `docker compose -f docker-compose.test.yml run --rm tests python build_index.py`
Expected: prints the two encoding lines, then `Done. 18xxx fatwas, 2xxxx chunks.`

- [ ] **Step 5: Run the integrity tests to verify they pass**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_index_integrity.py -v`
Expected: 10 passed

- [ ] **Step 6: Remove the superseded index files and commit**

```bash
git rm --cached data/index/fatwas_embeddings.npy
rm -f data/index/fatwas_embeddings.npy
git add build_index.py tests/test_index_integrity.py data/index/
git commit -m "feat: rebuild index with chunked two-field fp16 embeddings"
```

---

## Task 7: Two-field retrieval

**Files:**
- Modify: `app/retrieval.py`
- Create: `tests/test_retrieval_scoring.py`

**Interfaces:**
- Consumes: index artifacts from Task 6; `normalize_arabic` from Task 2.
- Produces:
  - `score_fatwas(query_vec, question_emb, answer_emb, answer_parent, alpha) -> np.ndarray` of shape `(n_fatwas,)`
  - `FatwaRetriever.search(question: str, top_k: int = 5) -> list[dict]` — unchanged
    public shape, now with `"similarity"` from the blended score.
  - `ALPHA` module constant, overridable via `RETRIEVAL_ALPHA`.

Scoring: `alpha * sim(q, question_vec) + (1 - alpha) * max over that fatwa's chunks of sim(q, chunk_vec)`.

- [ ] **Step 1: Write the failing scoring tests**

```python
# tests/test_retrieval_scoring.py
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
    q_emb = np.stack([_unit([0, 1])]).astype(np.float16)
    # Two chunks for fatwa 0; the second matches the query exactly.
    a_emb = np.stack([_unit([0, 1]), _unit([1, 0])]).astype(np.float16)
    parent = np.array([0, 0], dtype=np.int32)
    scores = score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=0.0)
    assert np.isclose(scores[0], 1.0, atol=1e-2)


def test_returns_one_score_per_fatwa():
    q_emb = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])]).astype(np.float16)
    a_emb = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])]).astype(np.float16)
    parent = np.array([0, 1, 2], dtype=np.int32)
    assert score_fatwas(_unit([1, 0]), q_emb, a_emb, parent, alpha=0.5).shape == (3,)


def test_fatwa_with_no_chunks_still_scored():
    # parent never references fatwa 1 -> its answer term must not be NaN/-inf
    q_emb = np.stack([_unit([1, 0]), _unit([0, 1])]).astype(np.float16)
    a_emb = np.stack([_unit([1, 0])]).astype(np.float16)
    parent = np.array([0], dtype=np.int32)
    scores = score_fatwas(_unit([0, 1]), q_emb, a_emb, parent, alpha=0.5)
    assert np.isfinite(scores).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_retrieval_scoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_fatwas'`

- [ ] **Step 3: Modify `app/retrieval.py`**

Add near the top, after the existing imports:

```python
import json
from .normalize import normalize_arabic

INDEX_DIR = Path("data/index")
QUESTION_EMB_PATH = INDEX_DIR / "question_emb.npy"
ANSWER_EMB_PATH = INDEX_DIR / "answer_emb.npy"
ANSWER_PARENT_PATH = INDEX_DIR / "answer_parent.npy"
META_PATH = INDEX_DIR / "fatwas_meta.parquet"
MANIFEST_PATH = INDEX_DIR / "index_manifest.json"

# Weight on question-to-question similarity. 81% of the old single-vector index
# was answer text while users send questions, which is what compressed scores.
ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.65"))


def score_fatwas(query_vec, question_emb, answer_emb, answer_parent, alpha=ALPHA):
    """Blend question similarity with max-pooled answer-chunk similarity.
    Returns one score per fatwa."""
    q_scores = question_emb.astype(np.float32) @ query_vec
    n = len(q_scores)

    a_scores = np.zeros(n, dtype=np.float32)
    if len(answer_emb):
        chunk_scores = answer_emb.astype(np.float32) @ query_vec
        np.maximum.at(a_scores, answer_parent, chunk_scores)

    return alpha * q_scores + (1.0 - alpha) * a_scores
```

Replace `FatwaRetriever.__init__` body with:

```python
        for p in (QUESTION_EMB_PATH, ANSWER_EMB_PATH, ANSWER_PARENT_PATH, META_PATH):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Run build_index.py first.")

        self.question_emb = np.load(QUESTION_EMB_PATH)
        self.answer_emb = np.load(ANSWER_EMB_PATH)
        self.answer_parent = np.load(ANSWER_PARENT_PATH)
        self.meta = pd.read_parquet(META_PATH)
        self.manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

        if len(self.question_emb) != len(self.meta):
            raise RuntimeError(
                f"question_emb rows ({len(self.question_emb)}) != meta rows ({len(self.meta)})"
            )
        if len(self.answer_parent) != len(self.answer_emb):
            raise RuntimeError("answer_parent length != answer_emb rows")

        self._model: Optional[BGEM3FlagModel] = None
        self._model_lock = threading.Lock()
```

Replace `embed` and `search` with:

```python
    def embed(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        outputs = model.encode([normalize_arabic(text)], batch_size=1, max_length=512)
        vecs = outputs["dense_vecs"].astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        mark_activity()
        query_vec = self.embed(question)[0]
        scores = score_fatwas(
            query_vec, self.question_emb, self.answer_emb, self.answer_parent
        )

        top_k = max(1, min(top_k, len(scores)))
        idx_part = np.argpartition(-scores, top_k - 1)[:top_k]
        idx_sorted = idx_part[np.argsort(-scores[idx_part])]

        results = []
        for idx in idx_sorted:
            row = self.meta.iloc[int(idx)]
            cats = row.get("categories")
            results.append({
                "id": int(row.get("id")),
                "question": str(row.get("question", "")),
                "title": str(row.get("title", "")),
                "answer": str(row.get("answer", "")),
                "link": str(row.get("link", "")),
                "categories": list(cats) if cats is not None else [],
                "similarity": float(scores[idx]),
            })
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_retrieval_scoring.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/retrieval.py tests/test_retrieval_scoring.py
git commit -m "feat: blend question and answer-chunk similarity in retrieval"
```

---

## Task 8: Golden dataset and retrieval eval

**Files:**
- Create: `eval/build_datasets.py`, `eval/run_retrieval_eval.py`, `eval/datasets/golden.jsonl`
- Create: `tests/test_eval_datasets.py`

**Interfaces:**
- Consumes: `eval.metrics.aggregate`, `FatwaRetriever`.
- Produces: `eval/datasets/golden.jsonl`, each line
  `{"query": str, "relevant_id": int, "kind": "verbatim" | "paraphrase"}`
- Produces: `eval/results/retrieval_<timestamp>.json`

Golden rows are generated from the corpus: 100 `verbatim` (the fatwa's own
question, normalized) and 100 `paraphrase` (question with leading interrogative
particles and filler stripped, to simulate how users actually type). Both are
mechanical, so the set is reproducible and needs no manual labelling.

- [ ] **Step 1: Write the failing dataset tests**

```python
# tests/test_eval_datasets.py
import json
from pathlib import Path

import pytest

GOLDEN = Path("eval/datasets/golden.jsonl")

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="golden set not built yet")


@pytest.fixture(scope="module")
def rows():
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_has_expected_size(rows):
    assert len(rows) == 200


def test_every_row_has_required_fields(rows):
    for r in rows:
        assert set(r) >= {"query", "relevant_id", "kind"}
        assert isinstance(r["relevant_id"], int)
        assert r["query"].strip()


def test_kinds_are_balanced(rows):
    kinds = [r["kind"] for r in rows]
    assert kinds.count("verbatim") == 100
    assert kinds.count("paraphrase") == 100


def test_relevant_ids_exist_in_meta(rows):
    import pandas as pd
    meta = pd.read_parquet("data/index/fatwas_meta.parquet")
    valid = set(meta["id"].tolist())
    assert {r["relevant_id"] for r in rows} <= valid


def test_no_duplicate_queries(rows):
    queries = [r["query"] for r in rows]
    assert len(queries) == len(set(queries))
```

- [ ] **Step 2: Run tests to verify they skip**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_eval_datasets.py -v`
Expected: 5 skipped

- [ ] **Step 3: Write `eval/build_datasets.py`**

```python
"""Generate the golden retrieval set from the corpus.

Two mechanical variants per sampled fatwa keep the set reproducible:
  verbatim   - the fatwa's own question, normalized
  paraphrase - the question stripped of leading interrogative particles and
               filler, approximating how a user actually types it
"""
import json
import re
from pathlib import Path

import pandas as pd

from app.normalize import normalize_arabic

META = Path("data/index/fatwas_meta.parquet")
OUT = Path("eval/datasets/golden.jsonl")
SEED = 20260808

_LEAD = re.compile(
    r"^(?:يقول|يسأل|السؤال|سؤال|أحسن الله إليكم|بارك الله فيكم|فضيلة الشيخ|"
    r"سماحة الشيخ|هذا السائل يقول|من|هل)\s*[:،]?\s*"
)


def paraphrase(question: str) -> str:
    q = normalize_arabic(question)
    prev = None
    while prev != q:                     # strip stacked openers
        prev = q
        q = _LEAD.sub("", q).strip()
    words = q.split()
    return " ".join(words[:18]) if words else normalize_arabic(question)


def main():
    meta = pd.read_parquet(META)
    long_enough = meta[meta["question"].astype(str).str.split().str.len() >= 8]
    sample = long_enough.sample(200, random_state=SEED)

    verbatim = sample.iloc[:100]
    para = sample.iloc[100:]

    rows, seen = [], set()
    for _, r in verbatim.iterrows():
        q = normalize_arabic(str(r["question"]))
        if q in seen:
            continue
        seen.add(q)
        rows.append({"query": q, "relevant_id": int(r["id"]), "kind": "verbatim"})

    for _, r in para.iterrows():
        q = paraphrase(str(r["question"]))
        if q in seen or not q.strip():
            continue
        seen.add(q)
        rows.append({"query": q, "relevant_id": int(r["id"]), "kind": "paraphrase"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `eval/run_retrieval_eval.py`**

```python
"""Score retrieval against the golden set and persist the result."""
import argparse
import json
from pathlib import Path

from app.retrieval import get_retriever
from eval.metrics import aggregate

DEFAULT_SET = Path("eval/datasets/golden.jsonl")
RESULTS_DIR = Path("eval/results")


def load_rows(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run(dataset: Path, top_k: int, tag: str):
    rows = load_rows(dataset)
    retriever = get_retriever()

    scored = []
    for i, row in enumerate(rows, 1):
        hits = retriever.search(row["query"], top_k=top_k)
        scored.append({
            "ranked_ids": [h["id"] for h in hits],
            "relevant_id": row["relevant_id"],
            "kind": row.get("kind", "unknown"),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    overall = aggregate(scored)
    by_kind = {
        kind: aggregate([s for s in scored if s["kind"] == kind])
        for kind in sorted({s["kind"] for s in scored})
    }

    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    for kind, m in by_kind.items():
        print(f"\n=== {kind} ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{tag}.json"
    out.write_text(json.dumps(
        {"dataset": str(dataset), "top_k": top_k, "overall": overall, "by_kind": by_kind},
        ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out}")
    return overall


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_SET)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--tag", default="latest")
    a = p.parse_args()
    run(a.dataset, a.top_k, a.tag)
```

- [ ] **Step 5: Build the golden set and run the tests**

Run:
```bash
docker compose -f docker-compose.test.yml run --rm tests python eval/build_datasets.py
docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_eval_datasets.py -v
```
Expected: `wrote 200 rows`, then 5 passed.

- [ ] **Step 6: Measure the baseline**

Run: `docker compose -f docker-compose.test.yml run --rm tests python eval/run_retrieval_eval.py --tag phase1`
Expected: `recall@1 >= 0.977` and `recall@5 >= 1.00` on the `verbatim` slice.
If `verbatim` recall@1 is below 0.977, **stop** — the rebuild regressed retrieval;
tune `RETRIEVAL_ALPHA` (Step 7) before proceeding.

- [ ] **Step 7: Tune alpha**

Run:
```bash
for a in 0.4 0.5 0.65 0.8 1.0; do
  echo "=== alpha=$a ==="
  docker compose -f docker-compose.test.yml run --rm -e RETRIEVAL_ALPHA=$a tests \
    python eval/run_retrieval_eval.py --tag "alpha_$a"
done
```
Pick the alpha with the best `paraphrase` MRR that does not reduce `verbatim`
recall@1 below 0.977. Set that value as the `ALPHA` default in `app/retrieval.py`.

- [ ] **Step 8: Commit**

```bash
git add eval/ tests/test_eval_datasets.py app/retrieval.py
git commit -m "feat: add golden retrieval eval and tune question/answer weighting"
```

---

## Task 9: Derived and adversarial datasets

**Files:**
- Create: `eval/datasets/derived.jsonl`, `eval/datasets/adversarial.jsonl`
- Modify: `tests/test_eval_datasets.py`

**Interfaces:**
- Produces:
  - `derived.jsonl` rows: `{"query": str, "expected_verdict": "derived", "supporting_ids": [int], "note": str, "source": str}`
  - `adversarial.jsonl` rows: `{"query": str, "expected_verdict": "abstain", "reason": "false_premise" | "out_of_scope" | "gibberish", "note": str}`

These are the only hand-curated datasets. `supporting_ids` must be verified to
exist in the corpus, and each derived row must genuinely be answerable from
those fatwas without inventing a ruling.

**Research constraint:** use Sunni sources only when confirming that a question
is answerable by derivation — `binbaz.org.sa`, `islamqa.info`, `islamweb.net`,
`dorar.net`, `alifta.gov.sa`. **Exclude Shia sources.** Record the consulted URL
in `source`.

- [ ] **Step 1: Curate 60 derived rows**

For each row: pick a real fatwa topic in the corpus, then write a question that
is *not* answered verbatim but is determined by it (a paraphrase in dialect, a
narrower case, or a direct application). Verify with a Sunni source that the
derivation is sound, and record `supporting_ids` by searching the corpus.

Minimum coverage: 20 dialectal paraphrases, 20 narrower cases, 20 applications.

- [ ] **Step 2: Curate 60 adversarial rows**

20 `false_premise` (including «اذا جا الحج ورمضان في نفس الوقت نصوم ولا نحج؟»),
20 `out_of_scope`, 20 `gibberish`.

- [ ] **Step 3: Add dataset validation tests**

```python
# append to tests/test_eval_datasets.py
DERIVED = Path("eval/datasets/derived.jsonl")
ADVERSARIAL = Path("eval/datasets/adversarial.jsonl")


def _load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.skipif(not DERIVED.exists(), reason="derived set not built yet")
def test_derived_set_shape():
    rows = _load(DERIVED)
    assert len(rows) == 60
    for r in rows:
        assert r["expected_verdict"] == "derived"
        assert r["query"].strip()
        assert isinstance(r["supporting_ids"], list) and r["supporting_ids"]
        assert r["source"].startswith("http")


@pytest.mark.skipif(not DERIVED.exists(), reason="derived set not built yet")
def test_derived_supporting_ids_exist():
    import pandas as pd
    meta = pd.read_parquet("data/index/fatwas_meta.parquet")
    valid = set(meta["id"].tolist())
    for r in _load(DERIVED):
        assert set(r["supporting_ids"]) <= valid, r["query"]


@pytest.mark.skipif(not ADVERSARIAL.exists(), reason="adversarial set not built yet")
def test_adversarial_set_shape():
    rows = _load(ADVERSARIAL)
    assert len(rows) == 60
    reasons = [r["reason"] for r in rows]
    for reason in ("false_premise", "out_of_scope", "gibberish"):
        assert reasons.count(reason) == 20
    for r in rows:
        assert r["expected_verdict"] == "abstain"


@pytest.mark.skipif(not ADVERSARIAL.exists(), reason="adversarial set not built yet")
def test_hajj_ramadan_case_is_present():
    queries = " ".join(r["query"] for r in _load(ADVERSARIAL))
    assert "الحج" in queries and "رمضان" in queries
```

- [ ] **Step 4: Run the dataset tests**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest tests/test_eval_datasets.py -v`
Expected: 9 passed

- [ ] **Step 5: Sanity-check retrieval reaches the supporting fatwas**

Run:
```bash
docker compose -f docker-compose.test.yml run --rm tests python -c "
import json
from app.retrieval import get_retriever
rows = [json.loads(l) for l in open('eval/datasets/derived.jsonl', encoding='utf-8')]
r = get_retriever()
hit = sum(bool(set(h['id'] for h in r.search(x['query'], top_k=8)) & set(x['supporting_ids'])) for x in rows)
print(f'derived rows whose supporting fatwa is in top-8: {hit}/{len(rows)}')
assert hit / len(rows) >= 0.80, 'retrieval cannot reach the supporting fatwas'
"
```
Expected: ≥ 80%. If lower, the derived set is asking for fatwas the corpus does
not contain — fix the dataset, not the threshold.

- [ ] **Step 6: Commit**

```bash
git add eval/datasets/derived.jsonl eval/datasets/adversarial.jsonl tests/test_eval_datasets.py
git commit -m "feat: add derived and adversarial eval datasets"
```

---

## Task 10: Full suite, docker smoke test, PR

**Files:**
- Modify: `docker-entrypoint.sh` (index filename check), `README.md`

- [ ] **Step 1: Update `docker-entrypoint.sh` to check the new artifact names**

Replace the index-existence check with:

```sh
if [ ! -f data/index/question_emb.npy ] || [ ! -f data/index/fatwas_meta.parquet ]; then
```

- [ ] **Step 2: Run the whole suite**

Run: `docker compose -f docker-compose.test.yml run --rm tests pytest -v`
Expected: all pass, none failed.

- [ ] **Step 3: Smoke-test the real app in docker**

Run:
```bash
docker compose up -d --build
curl -sf http://localhost:8000/health
curl -s -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"ما حكم صيام يوم عرفة لغير الحاج؟","top_k":5}' | head -c 800
docker compose down
```
Expected: `{"status":"ok"}` then a JSON response whose `related_fatwas` have
non-empty `categories` (proving the category fix reaches the API).

- [ ] **Step 4: Update `README.md`**

Correct the stale sections: the LLM is the HF Inference API (not local ALLaM/llama-cpp),
there is no Gradio `app.py`, and document the new index artifacts and
`RETRIEVAL_ALPHA`. Remove the `sdk: gradio` / `app_file: app.py` front-matter keys
that point at a file which does not exist.

- [ ] **Step 5: Commit and open the PR**

```bash
git add docker-entrypoint.sh README.md
git commit -m "docs: correct README and entrypoint for the rebuilt index"
git push -u origin feat/faithful-rag
gh pr create --base main --head feat/faithful-rag \
  --title "Phase 0+1: eval harness and index rebuild" \
  --body "$(cat <<'BODY'
## Summary
Builds the measurement harness and rebuilds the retrieval index.

- Adds Arabic normalization shared by index and query time
- Chunks long answers: 17.7% of fatwas were truncated at max_length=512, losing 19.6% of corpus tokens
- Splits question and answer vectors: the old single vector was 81% answer text while users send questions
- Drops 72 rows whose answer was the literal string "nan"
- Parses `categories` into real lists, so the API field is no longer always null
- Adds golden / derived / adversarial eval datasets and a retrieval eval harness

## Verification
- Full pytest suite green in Docker
- Retrieval metrics at or above the pre-change baseline (recall@1 0.977, recall@5 1.00, MRR 0.988)
- App smoke-tested in Docker: /health and /api/chat respond, categories populated
BODY
)"
```

- [ ] **Step 6: Merge**

```bash
gh pr merge --squash --delete-branch
```

---

## Self-Review Notes

**Spec coverage check.** Spec §7.3 data fixes → Task 5. §7.1 two-field → Tasks 6, 7.
§7.2 fp16 → Task 6. §9 retrieval metrics → Tasks 4, 8. §9 derived/adversarial sets →
Task 9. §1.1 corpus coverage → Task 6 (chunking).

**Deferred to later phases, by design:** the gate, verdicts, citations, prompts
(Phase 3); hybrid sparse retrieval, RRF, MMR (Phase 2); frontend (Phase 4);
rate limiting, CORS, logging (Phase 5). The faithfulness and abstention metrics
defined in spec §9 need the gate to exist, so they are built in Phase 3 against
the datasets Task 9 creates here.

**Known risk carried into execution:** Task 8 Step 6 can fail — the rebuilt index
may not reproduce the 0.977 baseline, because the baseline was measured on the old
single-vector index using questions that were part of the embedded text. Task 8
Step 7 (alpha tuning) is the remedy; if no alpha recovers it, the two-field design
needs revisiting before Phase 2.
