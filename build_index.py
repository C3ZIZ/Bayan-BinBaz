"""Build the retrieval index.

Two changes versus the original that materially affect quality:

  1. ``max_length`` was 512 while BGE-M3 supports 8192. Measured with the real
     tokenizer, that truncated 17.7% of fatwas and left 19.6% of all corpus
     tokens out of the index entirely. Answers are now chunked, so no content
     is dropped.
  2. The original embedded ``question + "\\n" + answer`` as a single vector, of
     which 81% of tokens were answer text — while users send short questions.
     That asymmetry compressed a perfect self-match to a median cosine of 0.815.
     Questions and answer chunks are embedded separately so question-to-question
     matching can dominate at query time (see app/retrieval.py::score_fatwas).

Vectors are L2-normalized and stored float16, halving the on-disk index with
negligible recall loss — it lives in git, so size matters.
"""

import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

from app.chunking import chunk_text
from app.normalize import NORMALIZE_VERSION, normalize_arabic

PROCESSED_PATH = Path("data/processed/fatwas.parquet")
INDEX_DIR = Path("data/index")
# Shard checkpoints live outside data/index so a partial build never looks
# like a usable index to the app or to git.
CACHE_DIR = Path("data/.index_cache")

EMB_MODEL = "BAAI/bge-m3"
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP = 100
# FlagEmbedding pads each batch to max_length rather than to the longest item
# in it, so an oversized max_length is paid on EVERY text. Chunking already
# bounds every input to CHUNK_MAX_TOKENS, so 512 covers all of them; using
# 2048 here made encoding roughly 4x slower for identical output.
MAX_LENGTH = int(os.getenv("BUILD_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BUILD_BATCH_SIZE", "32"))

# Measured on an 8-thread CPU: questions encode at ~2.8/s but answer chunks at
# ~0.92/s, so the ~23k answer chunks cost ~7 hours against ~1.9 hours for the
# questions. The answer-side dense pass is therefore optional. With it off,
# dense retrieval matches question-to-question (which is what the corpus
# asymmetry called for anyway) and BM25 covers the answer text lexically.
BUILD_ANSWER_INDEX = os.getenv("BUILD_ANSWER_INDEX", "1") == "1"
SHARD_SIZE = int(os.getenv("BUILD_SHARD_SIZE", "1024"))


def _encode(model, texts, label):
    """Encode in checkpointed shards.

    Encoding the whole corpus in one call is a ~2 hour operation whose partial
    work is unrecoverable: a single OOM or container restart loses everything.
    Each shard is written to CACHE_DIR as it completes and reused on a rerun, so
    a crash costs one shard rather than the whole build.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    total = len(texts)
    n_shards = (total + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"Encoding {total} {label} in {n_shards} shards of {SHARD_SIZE}...", flush=True)

    parts = []
    for shard in range(n_shards):
        path = CACHE_DIR / f"{label}_{shard:05d}.npy"
        if path.exists():
            parts.append(np.load(path))
            print(f"  [{shard + 1}/{n_shards}] cached", flush=True)
            continue

        chunk = texts[shard * SHARD_SIZE : (shard + 1) * SHARD_SIZE]
        started = time.time()
        out = model.encode(chunk, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
        vecs = out["dense_vecs"].astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs.astype(np.float16)

        np.save(path, vecs)
        parts.append(vecs)
        del out
        gc.collect()

        elapsed = time.time() - started
        done = shard + 1
        eta = elapsed * (n_shards - done) / 60
        print(
            f"  [{done}/{n_shards}] {len(chunk)} in {elapsed:.0f}s "
            f"({len(chunk) / elapsed:.1f}/s) — ETA {eta:.0f} min",
            flush=True,
        )

    return np.vstack(parts) if parts else np.zeros((0, 1024), dtype=np.float16)


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

    question_emb = _encode(model, questions, "questions")

    chunk_parent: list = []
    if BUILD_ANSWER_INDEX:
        chunk_texts = []
        for row_idx, answer in enumerate(df["answer"].fillna("").tolist()):
            pieces = chunk_text(
                normalize_arabic(answer),
                token_len,
                max_tokens=CHUNK_MAX_TOKENS,
                overlap_tokens=CHUNK_OVERLAP,
            )
            if not pieces:
                # Terse or empty answer: fall back to the question so every
                # fatwa still owns at least one chunk and stays reachable.
                pieces = [questions[row_idx] or " "]
            for piece in pieces:
                chunk_texts.append(piece)
                chunk_parent.append(row_idx)
        answer_emb = _encode(model, chunk_texts, "answer chunks")
    else:
        print("Skipping answer index (BUILD_ANSWER_INDEX=0).", flush=True)
        answer_emb = np.zeros((0, question_emb.shape[1]), dtype=np.float16)

    np.save(INDEX_DIR / "question_emb.npy", question_emb)
    np.save(INDEX_DIR / "answer_emb.npy", answer_emb)
    np.save(INDEX_DIR / "answer_parent.npy", np.asarray(chunk_parent, dtype=np.int32))

    meta_cols = [
        c for c in ["id", "question", "title", "answer", "link", "categories"]
        if c in df.columns
    ]
    df[meta_cols].to_parquet(INDEX_DIR / "fatwas_meta.parquet", index=False)

    (INDEX_DIR / "index_manifest.json").write_text(
        json.dumps(
            {
                "normalize_version": NORMALIZE_VERSION,
                "emb_model": EMB_MODEL,
                "max_length": MAX_LENGTH,
                "chunk_max_tokens": CHUNK_MAX_TOKENS,
                "chunk_overlap": CHUNK_OVERLAP,
                "n_fatwas": int(len(question_emb)),
                "n_chunks": int(len(answer_emb)),
                "dtype": "float16",
                "has_answer_index": bool(BUILD_ANSWER_INDEX),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Only clear the checkpoints once every artifact is safely written.
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print(f"Done. {len(question_emb)} fatwas, {len(answer_emb)} chunks.")


if __name__ == "__main__":
    main()
