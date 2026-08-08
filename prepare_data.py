"""Load the raw fatwa CSV, clean it, and write data/processed/fatwas.parquet.

Three defects fixed versus the original implementation:

  1. ``.astype(str)`` ran before ``dropna``, turning NaN into the string "nan"
     and defeating the null filter — 72 such rows reached the index and were
     embedded as real documents whose entire answer was the word "nan".
  2. ``categories`` stayed a stringified list ("['التفسير']"), so the API's
     category field was always None despite 100% coverage across 215 categories.
  3. Duplicate links were never collapsed, so top-k could return the same fatwa
     twice under different ids.
"""

import ast
import math
from pathlib import Path
from typing import Any, List

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "fatwas.parquet"

# Values that mean "missing" once a CSV round-trip has stringified them.
_NULLISH = {"", "nan", "none", "null", "na"}

_COLUMN_ALIASES = [
    ("question", ("question", "questions")),
    ("answer", ("answer", "answers")),
    ("title", ("title", "titles")),
    ("link", ("link", "links", "url")),
    ("categories", ("categories", "category")),
]


def parse_categories(value: Any) -> List[str]:
    """Turn "['التفسير']" (or an actual list) into ["التفسير"]. Never raises."""
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
    """Rename columns to the canonical schema, drop unusable rows, parse
    categories. Returns exactly the columns the index build expects."""
    cols = {c.lower(): c for c in df.columns}

    rename_map = {}
    for target, candidates in _COLUMN_ALIASES:
        for name in candidates:
            if name in cols:
                rename_map[cols[name]] = target
                break
    df = df.rename(columns=rename_map)

    for col in ("title", "link", "categories"):
        if col not in df.columns:
            df[col] = None

    # Drop nullish rows BEFORE any string coercion. Doing this after
    # .astype(str) is the original bug: NaN becomes "nan" and survives.
    keep = ~(df["question"].map(_is_nullish) | df["answer"].map(_is_nullish))
    df = df[keep].copy()

    df["categories"] = df["categories"].map(parse_categories)
    for col in ("question", "answer", "title", "link"):
        df[col] = df[col].fillna("").astype(str).str.strip()

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
            print(f"[SKIP] {path} (error while reading: {e})")
            continue

        norm = normalize_fatwa_table(df)
        if len(norm):
            print(f"[OK] Loaded {len(norm)} rows from {path}")
            tables.append(norm)

    if not tables:
        raise RuntimeError(
            "No usable data in data/raw.\n"
            "Expected columns: questions, answers, titles, links, categories."
        )

    df_all = pd.concat(tables, ignore_index=True)

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["question", "answer"])
    df_all = df_all[
        (df_all["link"] == "") | ~df_all.duplicated(subset=["link"], keep="first")
    ]
    print(f"[dedup] {before} -> {len(df_all)} rows")

    df_all = df_all.reset_index(drop=True)
    df_all.insert(0, "id", range(1, len(df_all) + 1))
    return df_all


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_fatwa_tables()
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} fatwas to {OUT_PATH}")


if __name__ == "__main__":
    main()
